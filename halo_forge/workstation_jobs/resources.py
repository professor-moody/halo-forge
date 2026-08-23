"""Honest, portable workstation capacity and work telemetry primitives.

All probes are best effort. An unavailable metric is represented by ``None``
and a diagnostic note; it is never converted to zero. The module has no hard
dependency on psutil or an accelerator runtime and performs no persistence.
"""

from __future__ import annotations

import math
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

TELEMETRY_SAMPLE_INTERVAL_SECONDS = 2.0
RAW_TELEMETRY_RETENTION_DAYS = 30
MINIMUM_DISK_RESERVE_BYTES = 20 * 1024**3
MINIMUM_RAM_RESERVE_BYTES = 1 * 1024**3
DEFAULT_RESERVE_FRACTION = 0.10


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _aware_datetime(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _optional_integer(value: Any, name: str) -> Optional[int]:
    return None if value is None else _integer(value, name)


def _optional_number(
    value: Any,
    name: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number or None")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must not exceed {maximum}")
    return result


def _frozen_metadata(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise TypeError("metadata must be a mapping")
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class DiskCapacity:
    path: str
    total_bytes: int
    used_bytes: int
    free_bytes: int

    def __post_init__(self) -> None:
        path = str(self.path).strip()
        if not path:
            raise ValueError("disk capacity path cannot be empty")
        total = _integer(self.total_bytes, "total_bytes")
        used = _integer(self.used_bytes, "used_bytes")
        free = _integer(self.free_bytes, "free_bytes")
        if used > total or free > total:
            raise ValueError("disk used/free bytes cannot exceed total bytes")
        object.__setattr__(self, "path", path)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "total_bytes": self.total_bytes,
            "used_bytes": self.used_bytes,
            "free_bytes": self.free_bytes,
        }


@dataclass(frozen=True)
class MemoryCapacity:
    total_bytes: int
    used_bytes: Optional[int]
    available_bytes: Optional[int]
    source: Optional[str] = None
    available_is_estimate: bool = False

    def __post_init__(self) -> None:
        total = _integer(self.total_bytes, "total_bytes", minimum=1)
        used = _optional_integer(self.used_bytes, "used_bytes")
        available = _optional_integer(self.available_bytes, "available_bytes")
        if used is not None and used > total:
            raise ValueError("memory used_bytes cannot exceed total_bytes")
        if available is not None and available > total:
            raise ValueError("memory available_bytes cannot exceed total_bytes")
        object.__setattr__(self, "total_bytes", total)
        object.__setattr__(self, "used_bytes", used)
        object.__setattr__(self, "available_bytes", available)
        source = None if self.source is None else str(self.source).strip() or None
        if not isinstance(self.available_is_estimate, bool):
            raise TypeError("available_is_estimate must be a boolean")
        if self.available_is_estimate and source is None:
            raise ValueError("estimated available memory requires a measurement source")
        object.__setattr__(self, "source", source)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_bytes": self.total_bytes,
            "used_bytes": self.used_bytes,
            "available_bytes": self.available_bytes,
            "source": self.source,
            "available_is_estimate": self.available_is_estimate,
        }


@dataclass(frozen=True)
class ProcessCapacity:
    pid: int
    rss_bytes: Optional[int] = None
    cpu_percent: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pid", _integer(self.pid, "pid", minimum=1))
        object.__setattr__(self, "rss_bytes", _optional_integer(self.rss_bytes, "rss_bytes"))
        # A process can exceed 100% on multicore systems, so there is no upper bound.
        object.__setattr__(
            self,
            "cpu_percent",
            _optional_number(self.cpu_percent, "cpu_percent", minimum=0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pid": self.pid,
            "rss_bytes": self.rss_bytes,
            "cpu_percent": self.cpu_percent,
        }


@dataclass(frozen=True)
class AcceleratorCapacity:
    backend: str
    device_name: Optional[str] = None
    gpu_percent: Optional[float] = None
    device_memory_used_bytes: Optional[int] = None
    device_memory_total_bytes: Optional[int] = None
    power_watts: Optional[float] = None
    temperature_c: Optional[float] = None
    note: Optional[str] = None

    def __post_init__(self) -> None:
        backend = str(self.backend).strip().lower()
        if not backend:
            raise ValueError("accelerator backend cannot be empty")
        used = _optional_integer(self.device_memory_used_bytes, "device_memory_used_bytes")
        total = _optional_integer(self.device_memory_total_bytes, "device_memory_total_bytes")
        if used is not None and total is not None and used > total:
            raise ValueError("device memory used bytes cannot exceed total bytes")
        object.__setattr__(self, "backend", backend)
        object.__setattr__(
            self, "device_name", None if self.device_name is None else str(self.device_name)
        )
        object.__setattr__(
            self,
            "gpu_percent",
            _optional_number(self.gpu_percent, "gpu_percent", minimum=0, maximum=100),
        )
        object.__setattr__(self, "device_memory_used_bytes", used)
        object.__setattr__(self, "device_memory_total_bytes", total)
        object.__setattr__(
            self,
            "power_watts",
            _optional_number(self.power_watts, "power_watts", minimum=0),
        )
        object.__setattr__(
            self, "temperature_c", _optional_number(self.temperature_c, "temperature_c")
        )
        object.__setattr__(self, "note", None if self.note is None else str(self.note))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "device_name": self.device_name,
            "gpu_percent": self.gpu_percent,
            "device_memory_used_bytes": self.device_memory_used_bytes,
            "device_memory_total_bytes": self.device_memory_total_bytes,
            "power_watts": self.power_watts,
            "temperature_c": self.temperature_c,
            "note": self.note,
        }


@dataclass(frozen=True)
class WorkstationCapacity:
    sampled_at: datetime
    disk: Optional[DiskCapacity]
    memory: Optional[MemoryCapacity]
    cpu_percent: Optional[float] = None
    process: Optional[ProcessCapacity] = None
    accelerator: Optional[AcceleratorCapacity] = None
    errors: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "sampled_at", _aware_datetime(self.sampled_at, "sampled_at"))
        if self.disk is not None and not isinstance(self.disk, DiskCapacity):
            raise TypeError("disk must be DiskCapacity or None")
        if self.memory is not None and not isinstance(self.memory, MemoryCapacity):
            raise TypeError("memory must be MemoryCapacity or None")
        if self.process is not None and not isinstance(self.process, ProcessCapacity):
            raise TypeError("process must be ProcessCapacity or None")
        if self.accelerator is not None and not isinstance(self.accelerator, AcceleratorCapacity):
            raise TypeError("accelerator must be AcceleratorCapacity or None")
        object.__setattr__(
            self,
            "cpu_percent",
            _optional_number(self.cpu_percent, "cpu_percent", minimum=0, maximum=100),
        )
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sampled_at": self.sampled_at.isoformat(),
            "disk": self.disk.to_dict() if self.disk else None,
            "memory": self.memory.to_dict() if self.memory else None,
            "cpu_percent": self.cpu_percent,
            "process": self.process.to_dict() if self.process else None,
            "accelerator": self.accelerator.to_dict() if self.accelerator else None,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class CapacityPreflightPolicy:
    minimum_disk_reserve_bytes: int = MINIMUM_DISK_RESERVE_BYTES
    minimum_ram_reserve_bytes: int = MINIMUM_RAM_RESERVE_BYTES
    disk_reserve_fraction: float = DEFAULT_RESERVE_FRACTION
    ram_reserve_fraction: float = DEFAULT_RESERVE_FRACTION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_disk_reserve_bytes",
            _integer(self.minimum_disk_reserve_bytes, "minimum_disk_reserve_bytes"),
        )
        object.__setattr__(
            self,
            "minimum_ram_reserve_bytes",
            _integer(self.minimum_ram_reserve_bytes, "minimum_ram_reserve_bytes"),
        )
        for name in ("disk_reserve_fraction", "ram_reserve_fraction"):
            value = _optional_number(getattr(self, name), name, minimum=0, maximum=1)
            if value is None:
                raise TypeError(f"{name} must be a number")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class CapacityPreflightResult:
    allowed: bool
    capacity_sufficient: bool
    overridden: bool
    override_reason: Optional[str]
    projected_disk_bytes: int
    projected_ram_bytes: int
    current_disk_free_bytes: Optional[int]
    projected_disk_free_bytes: Optional[int]
    required_disk_reserve_bytes: Optional[int]
    current_ram_available_bytes: Optional[int]
    projected_ram_available_bytes: Optional[int]
    required_ram_reserve_bytes: Optional[int]
    ram_available_is_estimate: bool
    blockers: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "capacity_sufficient": self.capacity_sufficient,
            "overridden": self.overridden,
            "override_reason": self.override_reason,
            "projected_disk_bytes": self.projected_disk_bytes,
            "projected_ram_bytes": self.projected_ram_bytes,
            "current_disk_free_bytes": self.current_disk_free_bytes,
            "projected_disk_free_bytes": self.projected_disk_free_bytes,
            "required_disk_reserve_bytes": self.required_disk_reserve_bytes,
            "current_ram_available_bytes": self.current_ram_available_bytes,
            "projected_ram_available_bytes": self.projected_ram_available_bytes,
            "required_ram_reserve_bytes": self.required_ram_reserve_bytes,
            "ram_available_is_estimate": self.ram_available_is_estimate,
            "blockers": list(self.blockers),
        }


def evaluate_capacity_preflight(
    capacity: WorkstationCapacity,
    *,
    projected_disk_bytes: int = 0,
    projected_ram_bytes: int = 0,
    override_reason: Optional[str] = None,
    policy: CapacityPreflightPolicy = CapacityPreflightPolicy(),
) -> CapacityPreflightResult:
    """Evaluate disk and RAM headroom, retaining explicit override provenance."""

    if not isinstance(capacity, WorkstationCapacity):
        raise TypeError("capacity must be WorkstationCapacity")
    projected_disk = _integer(projected_disk_bytes, "projected_disk_bytes")
    projected_ram = _integer(projected_ram_bytes, "projected_ram_bytes")
    if not isinstance(policy, CapacityPreflightPolicy):
        raise TypeError("policy must be CapacityPreflightPolicy")
    if override_reason is not None and not isinstance(override_reason, str):
        raise TypeError("override_reason must be a string or None")

    blockers = []
    disk_free = capacity.disk.free_bytes if capacity.disk else None
    disk_reserve = (
        max(
            policy.minimum_disk_reserve_bytes,
            math.ceil(capacity.disk.total_bytes * policy.disk_reserve_fraction),
        )
        if capacity.disk
        else None
    )
    projected_disk_free = None if disk_free is None else disk_free - projected_disk
    if projected_disk_free is None or disk_reserve is None:
        blockers.append("disk_capacity_unavailable")
    elif projected_disk_free < disk_reserve:
        blockers.append("insufficient_disk")

    ram_available = capacity.memory.available_bytes if capacity.memory else None
    ram_reserve = (
        max(
            policy.minimum_ram_reserve_bytes,
            math.ceil(capacity.memory.total_bytes * policy.ram_reserve_fraction),
        )
        if capacity.memory
        else None
    )
    projected_ram_available = None if ram_available is None else ram_available - projected_ram
    if projected_ram_available is None or ram_reserve is None:
        blockers.append("ram_capacity_unavailable")
    elif projected_ram_available < ram_reserve:
        blockers.append("insufficient_ram")

    native_pass = not blockers
    normalized_reason = None if override_reason is None else override_reason.strip() or None
    overridden = bool(blockers and normalized_reason)
    return CapacityPreflightResult(
        allowed=native_pass or overridden,
        capacity_sufficient=native_pass,
        overridden=overridden,
        override_reason=normalized_reason if overridden else None,
        projected_disk_bytes=projected_disk,
        projected_ram_bytes=projected_ram,
        current_disk_free_bytes=disk_free,
        projected_disk_free_bytes=projected_disk_free,
        required_disk_reserve_bytes=disk_reserve,
        current_ram_available_bytes=ram_available,
        projected_ram_available_bytes=projected_ram_available,
        required_ram_reserve_bytes=ram_reserve,
        ram_available_is_estimate=(
            capacity.memory.available_is_estimate if capacity.memory else False
        ),
        blockers=tuple(blockers),
    )


preflight_resources = evaluate_capacity_preflight


def _existing_disk_path(path: Path) -> Path:
    candidate = path.expanduser().resolve(strict=False)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _linux_memory_probe(path: Path = Path("/proc/meminfo")) -> Optional[MemoryCapacity]:
    try:
        values: Dict[str, int] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                key, raw = line.split(":", 1)
                amount = raw.strip().split()[0]
                values[key] = int(amount) * 1024
        total = values["MemTotal"]
        available = values.get("MemAvailable", values.get("MemFree"))
        return MemoryCapacity(
            total_bytes=total,
            used_bytes=None if available is None else total - available,
            available_bytes=available,
            source="proc_meminfo",
        )
    except (OSError, KeyError, ValueError, IndexError):
        return None


def _darwin_memory_probe() -> Optional[MemoryCapacity]:
    """Read macOS page counters, explicitly marking available RAM as estimated."""

    try:
        total_result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        pages_result = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if total_result.returncode != 0 or pages_result.returncode != 0:
            return None
        total = int(total_result.stdout.strip())
        page_size_match = re.search(r"page size of (\d+) bytes", pages_result.stdout)
        if page_size_match:
            page_size = int(page_size_match.group(1))
        else:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
        pages: Dict[str, int] = {}
        for line in pages_result.stdout.splitlines():
            if ":" not in line:
                continue
            key, raw = line.split(":", 1)
            match = re.search(r"(\d+)", raw.replace(".", ""))
            if match:
                pages[key.strip()] = int(match.group(1))
        # Inactive and speculative pages are reclaimable, but macOS can change
        # their status between samples. Preserve that uncertainty in the type.
        available_pages = sum(
            pages.get(key, 0) for key in ("Pages free", "Pages inactive", "Pages speculative")
        )
        if available_pages <= 0:
            return MemoryCapacity(
                total_bytes=total,
                used_bytes=None,
                available_bytes=None,
                source="darwin_vm_stat",
                available_is_estimate=True,
            )
        available = min(total, available_pages * page_size)
        return MemoryCapacity(
            total_bytes=total,
            used_bytes=total - available,
            available_bytes=available,
            source="darwin_vm_stat",
            available_is_estimate=True,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _sysconf_memory_probe() -> Optional[MemoryCapacity]:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total = int(os.sysconf("SC_PHYS_PAGES")) * page_size
        available = int(os.sysconf("SC_AVPHYS_PAGES")) * page_size
        return MemoryCapacity(
            total_bytes=total,
            used_bytes=total - available,
            available_bytes=available,
            source="sysconf",
        )
    except (AttributeError, KeyError, OSError, ValueError):
        return None


def _default_memory_probe() -> Optional[MemoryCapacity]:
    try:
        import psutil  # type: ignore[import-not-found]

        memory = psutil.virtual_memory()
        return MemoryCapacity(
            total_bytes=int(memory.total),
            used_bytes=int(memory.used),
            available_bytes=int(memory.available),
            source="psutil",
        )
    except Exception:
        pass

    if platform.system() == "Linux":
        memory = _linux_memory_probe()
        if memory is not None:
            return memory
    if platform.system() == "Darwin":
        memory = _darwin_memory_probe()
        if memory is not None:
            return memory
    return _sysconf_memory_probe()


def _default_cpu_probe() -> Optional[float]:
    try:
        import psutil  # type: ignore[import-not-found]

        return float(psutil.cpu_percent(interval=None))
    except Exception:
        return None


def _default_process_probe(pid: int) -> Optional[ProcessCapacity]:
    try:
        import psutil  # type: ignore[import-not-found]

        process = psutil.Process(pid)
        return ProcessCapacity(
            pid=pid,
            rss_bytes=int(process.memory_info().rss),
            cpu_percent=float(process.cpu_percent(interval=None)),
        )
    except Exception:
        pass
    # Linux fallback can report RSS but not an honest instantaneous CPU value.
    try:
        fields = Path(f"/proc/{pid}/statm").read_text(encoding="utf-8").split()
        rss = int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
        return ProcessCapacity(pid=pid, rss_bytes=rss, cpu_percent=None)
    except (OSError, ValueError, IndexError):
        return None


def _telemetry_accelerator_probe() -> Optional[AcceleratorCapacity]:
    try:
        from halo_forge.telemetry.registry import get_telemetry_provider

        sample = get_telemetry_provider().sample()
    except Exception:
        return None

    def gb_to_bytes(value: Any) -> Optional[int]:
        if value is None:
            return None
        return int(round(float(value) * 1_000_000_000))

    return AcceleratorCapacity(
        backend=str(sample.backend),
        device_name=sample.device_name,
        gpu_percent=sample.gpu_util_percent,
        device_memory_used_bytes=gb_to_bytes(sample.vram_used_gb),
        device_memory_total_bytes=gb_to_bytes(sample.vram_total_gb),
        power_watts=sample.power_watts,
        temperature_c=sample.temp_celsius,
        note=sample.note,
    )


def _coerce_probe_result(value: Any, expected_type: type, name: str) -> Any:
    if value is None or isinstance(value, expected_type):
        return value
    if isinstance(value, Mapping):
        return expected_type(**dict(value))
    raise TypeError(f"{name} probe must return {expected_type.__name__}, a mapping, or None")


def sample_workstation_capacity(
    path: str | Path,
    *,
    pid: Optional[int] = None,
    memory_probe: Optional[Callable[[], MemoryCapacity | Mapping[str, Any] | None]] = None,
    cpu_probe: Optional[Callable[[], Optional[float]]] = None,
    process_probe: Optional[Callable[[int], ProcessCapacity | Mapping[str, Any] | None]] = None,
    accelerator_probe: Optional[
        Callable[[], AcceleratorCapacity | Mapping[str, Any] | None]
    ] = None,
    include_accelerator: bool = True,
    now: Optional[datetime] = None,
) -> WorkstationCapacity:
    """Sample local capacity without requiring psutil or a GPU runtime."""

    sampled_at = _aware_datetime(now or _utc_now(), "now")
    if not isinstance(include_accelerator, bool):
        raise TypeError("include_accelerator must be a boolean")
    requested_path = Path(path).expanduser().resolve(strict=False)
    errors = []
    disk = None
    try:
        usage = shutil.disk_usage(_existing_disk_path(requested_path))
        disk = DiskCapacity(
            path=str(requested_path),
            total_bytes=int(usage.total),
            used_bytes=int(usage.used),
            free_bytes=int(usage.free),
        )
    except (OSError, ValueError) as exc:
        errors.append(f"disk: {type(exc).__name__}: {exc}")

    memory = None
    try:
        memory = _coerce_probe_result(
            (memory_probe or _default_memory_probe)(), MemoryCapacity, "memory"
        )
    except Exception as exc:
        errors.append(f"memory: {type(exc).__name__}: {exc}")
    if memory is None:
        errors.append("memory: unavailable")

    cpu_percent = None
    try:
        cpu_percent = _optional_number(
            (cpu_probe or _default_cpu_probe)(),
            "cpu_percent",
            minimum=0,
            maximum=100,
        )
    except Exception as exc:
        errors.append(f"cpu: {type(exc).__name__}: {exc}")

    process = None
    if pid is not None:
        normalized_pid = _integer(pid, "pid", minimum=1)
        try:
            sampled_process = _coerce_probe_result(
                (process_probe or _default_process_probe)(normalized_pid),
                ProcessCapacity,
                "process",
            )
            if sampled_process is not None and sampled_process.pid != normalized_pid:
                raise ValueError("process probe returned metrics for a different pid")
            process = sampled_process
        except Exception as exc:
            errors.append(f"process: {type(exc).__name__}: {exc}")
        if process is None:
            errors.append("process: unavailable")

    accelerator = None
    if include_accelerator:
        try:
            accelerator = _coerce_probe_result(
                (accelerator_probe or _telemetry_accelerator_probe)(),
                AcceleratorCapacity,
                "accelerator",
            )
        except Exception as exc:
            errors.append(f"accelerator: {type(exc).__name__}: {exc}")

    return WorkstationCapacity(
        sampled_at=sampled_at,
        disk=disk,
        memory=memory,
        cpu_percent=cpu_percent,
        process=process,
        accelerator=accelerator,
        errors=tuple(errors),
    )


@dataclass(frozen=True)
class WorkstationTelemetrySample:
    """The persistent two-second sample shape used for active work."""

    work_item_id: str
    attempt_id: Optional[str]
    sampled_at: datetime
    interval_seconds: float = TELEMETRY_SAMPLE_INTERVAL_SECONDS
    cpu_percent: Optional[float] = None
    process_rss_bytes: Optional[int] = None
    system_memory_used_bytes: Optional[int] = None
    system_memory_total_bytes: Optional[int] = None
    gpu_percent: Optional[float] = None
    device_memory_used_bytes: Optional[int] = None
    device_memory_total_bytes: Optional[int] = None
    power_watts: Optional[float] = None
    temperature_c: Optional[float] = None
    throughput_tokens_per_second: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        work_item_id = str(self.work_item_id).strip()
        if not work_item_id:
            raise ValueError("work_item_id cannot be empty")
        attempt_id = None if self.attempt_id is None else str(self.attempt_id).strip() or None
        sampled_at = _aware_datetime(self.sampled_at, "sampled_at")
        interval = _optional_number(self.interval_seconds, "interval_seconds", minimum=0)
        if interval != TELEMETRY_SAMPLE_INTERVAL_SECONDS:
            raise ValueError(
                f"work telemetry interval must be {TELEMETRY_SAMPLE_INTERVAL_SECONDS:g} seconds"
            )
        object.__setattr__(self, "work_item_id", work_item_id)
        object.__setattr__(self, "attempt_id", attempt_id)
        object.__setattr__(self, "sampled_at", sampled_at)
        object.__setattr__(self, "interval_seconds", interval)
        for name in (
            "process_rss_bytes",
            "system_memory_used_bytes",
            "system_memory_total_bytes",
            "device_memory_used_bytes",
            "device_memory_total_bytes",
        ):
            object.__setattr__(self, name, _optional_integer(getattr(self, name), name))
        if (
            self.system_memory_used_bytes is not None
            and self.system_memory_total_bytes is not None
            and self.system_memory_used_bytes > self.system_memory_total_bytes
        ):
            raise ValueError("system memory used bytes cannot exceed total bytes")
        if (
            self.device_memory_used_bytes is not None
            and self.device_memory_total_bytes is not None
            and self.device_memory_used_bytes > self.device_memory_total_bytes
        ):
            raise ValueError("device memory used bytes cannot exceed total bytes")
        for name in ("cpu_percent", "gpu_percent"):
            object.__setattr__(
                self,
                name,
                _optional_number(getattr(self, name), name, minimum=0, maximum=100),
            )
        for name in ("power_watts", "throughput_tokens_per_second"):
            object.__setattr__(
                self,
                name,
                _optional_number(getattr(self, name), name, minimum=0),
            )
        object.__setattr__(
            self, "temperature_c", _optional_number(self.temperature_c, "temperature_c")
        )
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata))

    @classmethod
    def from_capacity(
        cls,
        work_item_id: str,
        capacity: WorkstationCapacity,
        *,
        attempt_id: Optional[str] = None,
        throughput_tokens_per_second: Optional[float] = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "WorkstationTelemetrySample":
        if not isinstance(capacity, WorkstationCapacity):
            raise TypeError("capacity must be WorkstationCapacity")
        combined_metadata = dict(metadata or {})
        if capacity.memory:
            combined_metadata["memory_source"] = capacity.memory.source
            combined_metadata["memory_available_is_estimate"] = (
                capacity.memory.available_is_estimate
            )
        if capacity.accelerator:
            combined_metadata["accelerator_backend"] = capacity.accelerator.backend
            combined_metadata["accelerator_device_name"] = capacity.accelerator.device_name
            combined_metadata["accelerator_note"] = capacity.accelerator.note
        if capacity.errors:
            combined_metadata["capacity_errors"] = list(capacity.errors)
        return cls(
            work_item_id=work_item_id,
            attempt_id=attempt_id,
            sampled_at=capacity.sampled_at,
            cpu_percent=capacity.cpu_percent,
            process_rss_bytes=capacity.process.rss_bytes if capacity.process else None,
            system_memory_used_bytes=capacity.memory.used_bytes if capacity.memory else None,
            system_memory_total_bytes=capacity.memory.total_bytes if capacity.memory else None,
            gpu_percent=capacity.accelerator.gpu_percent if capacity.accelerator else None,
            device_memory_used_bytes=(
                capacity.accelerator.device_memory_used_bytes if capacity.accelerator else None
            ),
            device_memory_total_bytes=(
                capacity.accelerator.device_memory_total_bytes if capacity.accelerator else None
            ),
            power_watts=capacity.accelerator.power_watts if capacity.accelerator else None,
            temperature_c=(capacity.accelerator.temperature_c if capacity.accelerator else None),
            throughput_tokens_per_second=throughput_tokens_per_second,
            metadata=combined_metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "work_item_id": self.work_item_id,
            "attempt_id": self.attempt_id,
            "sampled_at": self.sampled_at.isoformat(),
            "interval_seconds": self.interval_seconds,
            "cpu_percent": self.cpu_percent,
            "process_rss_bytes": self.process_rss_bytes,
            "system_memory_used_bytes": self.system_memory_used_bytes,
            "system_memory_total_bytes": self.system_memory_total_bytes,
            "gpu_percent": self.gpu_percent,
            "device_memory_used_bytes": self.device_memory_used_bytes,
            "device_memory_total_bytes": self.device_memory_total_bytes,
            "power_watts": self.power_watts,
            "temperature_c": self.temperature_c,
            "throughput_tokens_per_second": self.throughput_tokens_per_second,
            "metadata": dict(self.metadata),
        }


_ROLLUP_METRICS = (
    "cpu_percent",
    "process_rss_bytes",
    "system_memory_used_bytes",
    "system_memory_total_bytes",
    "gpu_percent",
    "device_memory_used_bytes",
    "device_memory_total_bytes",
    "power_watts",
    "temperature_c",
    "throughput_tokens_per_second",
)


@dataclass(frozen=True)
class TelemetryMetricAggregate:
    metric: str
    count: int
    minimum: Optional[float]
    maximum: Optional[float]
    mean: Optional[float]

    def __post_init__(self) -> None:
        metric = str(self.metric).strip()
        if not metric:
            raise ValueError("telemetry aggregate metric cannot be empty")
        if metric not in _ROLLUP_METRICS:
            raise ValueError(f"unsupported telemetry aggregate metric {metric!r}")
        count = _integer(self.count, "count")
        values = (self.minimum, self.maximum, self.mean)
        if count == 0 and any(value is not None for value in values):
            raise ValueError("an unavailable metric cannot contain aggregate values")
        if count > 0 and any(value is None for value in values):
            raise ValueError("an available metric requires complete aggregate values")
        normalized = tuple(_optional_number(value, metric) for value in values)
        minimum, maximum, mean = normalized
        if count > 0 and not (minimum <= mean <= maximum):
            raise ValueError("telemetry aggregate mean must fall between minimum and maximum")
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "count", count)
        for name, value in zip(("minimum", "maximum", "mean"), normalized):
            object.__setattr__(self, name, value)

    @classmethod
    def from_values(cls, metric: str, values: Sequence[float]) -> "TelemetryMetricAggregate":
        normalized = tuple(float(value) for value in values)
        if not normalized:
            return cls(metric, 0, None, None, None)
        return cls(
            metric=metric,
            count=len(normalized),
            minimum=min(normalized),
            maximum=max(normalized),
            mean=sum(normalized) / len(normalized),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "count": self.count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
        }


@dataclass(frozen=True)
class TelemetryRollup:
    work_item_id: str
    attempt_id: Optional[str]
    sample_count: int
    started_at: datetime
    ended_at: datetime
    metrics: Tuple[TelemetryMetricAggregate, ...]

    def __post_init__(self) -> None:
        work_item_id = str(self.work_item_id).strip()
        if not work_item_id:
            raise ValueError("work_item_id cannot be empty")
        attempt_id = None if self.attempt_id is None else str(self.attempt_id).strip() or None
        sample_count = _integer(self.sample_count, "sample_count", minimum=1)
        started_at = _aware_datetime(self.started_at, "started_at")
        ended_at = _aware_datetime(self.ended_at, "ended_at")
        if ended_at < started_at:
            raise ValueError("telemetry rollup cannot end before it starts")
        metrics = tuple(self.metrics)
        if any(not isinstance(metric, TelemetryMetricAggregate) for metric in metrics):
            raise TypeError("metrics must contain TelemetryMetricAggregate values")
        names = [metric.metric for metric in metrics]
        if set(names) != set(_ROLLUP_METRICS) or len(names) != len(_ROLLUP_METRICS):
            raise ValueError("telemetry rollup must contain each supported metric exactly once")
        if any(metric.count > sample_count for metric in metrics):
            raise ValueError("telemetry metric coverage cannot exceed sample_count")
        object.__setattr__(self, "work_item_id", work_item_id)
        object.__setattr__(self, "attempt_id", attempt_id)
        object.__setattr__(self, "sample_count", sample_count)
        object.__setattr__(self, "started_at", started_at)
        object.__setattr__(self, "ended_at", ended_at)
        object.__setattr__(self, "metrics", metrics)

    def metric(self, name: str) -> TelemetryMetricAggregate:
        for metric in self.metrics:
            if metric.metric == name:
                return metric
        raise KeyError(name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "work_item_id": self.work_item_id,
            "attempt_id": self.attempt_id,
            "sample_count": self.sample_count,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat(),
            "metrics": {metric.metric: metric.to_dict() for metric in self.metrics},
        }


def aggregate_telemetry(samples: Sequence[WorkstationTelemetrySample]) -> TelemetryRollup:
    """Aggregate samples for one work item, or one attempt when IDs agree."""

    unvalidated = tuple(samples)
    if not unvalidated:
        raise ValueError("at least one telemetry sample is required")
    if any(not isinstance(sample, WorkstationTelemetrySample) for sample in unvalidated):
        raise TypeError("samples must contain WorkstationTelemetrySample values")
    normalized = tuple(sorted(unvalidated, key=lambda sample: sample.sampled_at))
    work_ids = {sample.work_item_id for sample in normalized}
    if len(work_ids) != 1:
        raise ValueError("telemetry rollups cannot mix work items")
    attempt_ids = {sample.attempt_id for sample in normalized}
    attempt_id = next(iter(attempt_ids)) if len(attempt_ids) == 1 else None
    metrics = tuple(
        TelemetryMetricAggregate.from_values(
            name,
            [
                float(getattr(sample, name))
                for sample in normalized
                if getattr(sample, name) is not None
            ],
        )
        for name in _ROLLUP_METRICS
    )
    return TelemetryRollup(
        work_item_id=normalized[0].work_item_id,
        attempt_id=attempt_id,
        sample_count=len(normalized),
        started_at=normalized[0].sampled_at,
        ended_at=normalized[-1].sampled_at,
        metrics=metrics,
    )


def aggregate_telemetry_by_attempt(
    samples: Sequence[WorkstationTelemetrySample],
) -> Tuple[TelemetryRollup, ...]:
    """Partition and aggregate samples deterministically by attempt identity."""

    grouped: Dict[Tuple[str, Optional[str]], list[WorkstationTelemetrySample]] = {}
    for sample in samples:
        if not isinstance(sample, WorkstationTelemetrySample):
            raise TypeError("samples must contain WorkstationTelemetrySample values")
        grouped.setdefault((sample.work_item_id, sample.attempt_id), []).append(sample)
    return tuple(
        aggregate_telemetry(grouped[key])
        for key in sorted(grouped, key=lambda value: (value[0], value[1] or ""))
    )


def raw_telemetry_retention_cutoff(
    *,
    now: Optional[datetime] = None,
    retention_days: int = RAW_TELEMETRY_RETENTION_DAYS,
) -> datetime:
    """Return the exclusive deletion cutoff for raw telemetry samples."""

    normalized_now = _aware_datetime(now or _utc_now(), "now")
    days = _integer(retention_days, "retention_days", minimum=1)
    return normalized_now - timedelta(days=days)


__all__ = [
    "AcceleratorCapacity",
    "CapacityPreflightPolicy",
    "CapacityPreflightResult",
    "DEFAULT_RESERVE_FRACTION",
    "DiskCapacity",
    "MINIMUM_DISK_RESERVE_BYTES",
    "MINIMUM_RAM_RESERVE_BYTES",
    "MemoryCapacity",
    "ProcessCapacity",
    "RAW_TELEMETRY_RETENTION_DAYS",
    "TELEMETRY_SAMPLE_INTERVAL_SECONDS",
    "TelemetryMetricAggregate",
    "TelemetryRollup",
    "WorkstationCapacity",
    "WorkstationTelemetrySample",
    "aggregate_telemetry",
    "aggregate_telemetry_by_attempt",
    "evaluate_capacity_preflight",
    "preflight_resources",
    "raw_telemetry_retention_cutoff",
    "sample_workstation_capacity",
]
