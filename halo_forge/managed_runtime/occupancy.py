"""Conservative external accelerator occupancy probes.

Missing tools, malformed output, and permission failures are *unknown*, never
zero. Only `/dev/kfd` owners count for ROCm; display-only `/dev/dri` users do
not block compute.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from .models import AcceleratorAvailability, ExternalAcceleratorOwner


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _process_name(pid: int) -> str:
    try:
        value = Path(f"/proc/{pid}/comm").read_text(encoding="utf-8").strip()
        return Path(value).name or "external process"
    except OSError:
        return "external process"


def _elapsed(pid: int) -> Optional[int]:
    try:
        from halo_forge.workstation_jobs.scheduler import process_start_time

        started = process_start_time(pid)
        if started is None:
            return None
        # Linux /proc start ticks are not epoch seconds. psutil is normally
        # installed on managed hosts; do not fabricate elapsed time otherwise.
        if started < 1_000_000_000:
            return None
        return max(0, int(time.time() - started))
    except Exception:
        return None


def _process_tree(seed: int) -> set[int]:
    values = {int(seed)}
    changed = True
    while changed:
        changed = False
        for entry in Path("/proc").glob("[0-9]*/stat"):
            try:
                pid = int(entry.parent.name)
                fields = entry.read_text(encoding="utf-8").rsplit(")", 1)[1].split()
                ppid = int(fields[1])
            except (OSError, ValueError, IndexError):
                continue
            if ppid in values and pid not in values:
                values.add(pid)
                changed = True
    # Also exclude parents so a worker launched from the dashboard is not
    # mistaken for an unrelated owner when inspecting inherited descriptors.
    current = int(seed)
    while current > 1:
        try:
            fields = Path(f"/proc/{current}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
            current = int(fields[1])
        except (OSError, ValueError, IndexError):
            break
        values.add(current)
    return values


def _kfd_owners(excluded: set[int]) -> tuple[ExternalAcceleratorOwner, ...]:
    owners: list[ExternalAcceleratorOwner] = []
    for process in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(process.name)
        except ValueError:
            continue
        if pid in excluded:
            continue
        try:
            descriptors = process.joinpath("fd").iterdir()
            has_kfd = any(os.readlink(str(fd)) == "/dev/kfd" for fd in descriptors)
        except (OSError, PermissionError):
            continue
        if has_kfd:
            owners.append(
                ExternalAcceleratorOwner(pid, _process_name(pid), _elapsed(pid), device="/dev/kfd")
            )
    return tuple(sorted(owners, key=lambda value: value.pid or 0))


def _run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv), check=False, capture_output=True, text=True, timeout=10, shell=False
    )


def probe_rocm(
    *,
    excluded_pids: Optional[Iterable[int]] = None,
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = _run,
) -> AcceleratorAvailability:
    excluded = set(excluded_pids or ()) | _process_tree(os.getpid())
    owners = _kfd_owners(excluded)
    try:
        completed = runner(("rocm-smi", "--showuse", "--json"))
    except Exception as exc:
        return AcceleratorAvailability("rocm", "unknown", _now(), None, owners, f"rocm-smi unavailable: {exc}")
    if completed.returncode != 0:
        return AcceleratorAvailability("rocm", "unknown", _now(), None, owners, "rocm-smi could not inspect accelerator occupancy")
    try:
        payload = json.loads(completed.stdout)
        values: list[float] = []
        for device in payload.values():
            if not isinstance(device, dict):
                continue
            raw = next((value for key, value in device.items() if "GPU use" in str(key)), None)
            if raw is None:
                continue
            values.append(float(str(raw).replace("%", "").strip()))
        if not values:
            raise ValueError("GPU use missing")
        utilization = max(values)
    except (TypeError, ValueError, json.JSONDecodeError):
        return AcceleratorAvailability("rocm", "unknown", _now(), None, owners, "rocm-smi returned occupancy data Halo Forge could not verify")
    state = "busy" if owners or utilization > 1.0 else "idle"
    reason = "external compute owns the accelerator" if owners else ("accelerator is active without a visible owner" if state == "busy" else None)
    return AcceleratorAvailability("rocm", state, _now(), utilization, owners, reason, {"device_count": len(values)})


def probe_cuda(
    *,
    excluded_pids: Optional[Iterable[int]] = None,
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = _run,
) -> AcceleratorAvailability:
    excluded = set(excluded_pids or ()) | _process_tree(os.getpid())
    argv = (
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    )
    try:
        completed = runner(argv)
    except Exception as exc:
        return AcceleratorAvailability("cuda", "unknown", _now(), None, (), f"nvidia-smi unavailable: {exc}")
    if completed.returncode != 0:
        return AcceleratorAvailability("cuda", "unknown", _now(), None, (), "nvidia-smi could not inspect compute processes")
    owners: list[ExternalAcceleratorOwner] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3:
            return AcceleratorAvailability("cuda", "unknown", _now(), None, tuple(owners), "nvidia-smi returned malformed compute-process data")
        try:
            pid = int(parts[0])
            memory = int(float(parts[2])) * 1024 * 1024
        except ValueError:
            return AcceleratorAvailability("cuda", "unknown", _now(), None, tuple(owners), "nvidia-smi returned malformed compute-process data")
        if pid not in excluded:
            owners.append(ExternalAcceleratorOwner(pid, Path(parts[1]).name, _elapsed(pid), memory_bytes=memory))
    return AcceleratorAvailability("cuda", "busy" if owners else "idle", _now(), None, tuple(owners), "external compute owns the accelerator" if owners else None)


def probe_accelerator(family: str, **kwargs: object) -> AcceleratorAvailability:
    value = str(family).lower()
    if value == "rocm":
        return probe_rocm(**kwargs)
    if value == "cuda":
        return probe_cuda(**kwargs)
    return AcceleratorAvailability(value, "unknown", _now(), None, (), f"no occupancy probe is registered for {value}")


def wait_for_stable_idle(
    family: str,
    *,
    samples: int = 3,
    interval_seconds: float = 2.0,
    probe: Callable[[str], AcceleratorAvailability] = probe_accelerator,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[bool, tuple[AcceleratorAvailability, ...]]:
    evidence: list[AcceleratorAvailability] = []
    for index in range(max(1, int(samples))):
        current = probe(family)
        evidence.append(current)
        if not current.idle:
            return False, tuple(evidence)
        if index + 1 < samples:
            sleeper(max(0.0, float(interval_seconds)))
    return True, tuple(evidence)

