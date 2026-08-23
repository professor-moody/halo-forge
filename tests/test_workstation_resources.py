from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from halo_forge.workstation_jobs import resources
from halo_forge.workstation_jobs.resources import (
    AcceleratorCapacity,
    CapacityPreflightPolicy,
    DiskCapacity,
    MemoryCapacity,
    ProcessCapacity,
    WorkstationCapacity,
    WorkstationTelemetrySample,
    aggregate_telemetry,
    aggregate_telemetry_by_attempt,
    evaluate_capacity_preflight,
    raw_telemetry_retention_cutoff,
    sample_workstation_capacity,
)

GIB = 1024**3
NOW = datetime(2026, 7, 14, 18, 30, tzinfo=timezone.utc)


def _capacity(
    *,
    disk_free: int = 40 * GIB,
    ram_available: int | None = 16 * GIB,
) -> WorkstationCapacity:
    return WorkstationCapacity(
        sampled_at=NOW,
        disk=DiskCapacity(
            path="/tmp/lab",
            total_bytes=100 * GIB,
            used_bytes=100 * GIB - disk_free,
            free_bytes=disk_free,
        ),
        memory=MemoryCapacity(
            total_bytes=32 * GIB,
            used_bytes=None if ram_available is None else 32 * GIB - ram_available,
            available_bytes=ram_available,
        ),
        cpu_percent=12.5,
        process=ProcessCapacity(pid=123, rss_bytes=2 * GIB, cpu_percent=150),
        accelerator=AcceleratorCapacity(
            backend="mlx",
            device_name="Apple Test",
            gpu_percent=None,
            device_memory_used_bytes=4 * GIB,
            device_memory_total_bytes=32 * GIB,
            power_watts=None,
            temperature_c=None,
            note="limited metrics",
        ),
    )


def test_capacity_sampling_combines_injected_portable_probes(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        resources.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=100 * GIB, used=60 * GIB, free=40 * GIB),
    )
    capacity = sample_workstation_capacity(
        tmp_path / "not-created-yet" / "artifacts",
        pid=321,
        memory_probe=lambda: MemoryCapacity(32 * GIB, 20 * GIB, 12 * GIB),
        cpu_probe=lambda: 17.5,
        process_probe=lambda pid: {"pid": pid, "rss_bytes": 3 * GIB, "cpu_percent": 125},
        accelerator_probe=lambda: {
            "backend": "cuda",
            "device_name": "Test GPU",
            "gpu_percent": 75,
            "device_memory_used_bytes": 8 * GIB,
            "device_memory_total_bytes": 16 * GIB,
            "power_watts": 200,
            "temperature_c": 62,
        },
        now=NOW,
    )

    assert capacity.sampled_at == NOW
    assert capacity.disk.free_bytes == 40 * GIB
    assert capacity.disk.path.endswith("not-created-yet/artifacts")
    assert capacity.memory.available_bytes == 12 * GIB
    assert capacity.process.pid == 321
    assert capacity.process.cpu_percent == 125
    assert capacity.accelerator.gpu_percent == 75
    assert capacity.accelerator.device_memory_total_bytes == 16 * GIB
    assert capacity.errors == ()
    assert capacity.to_dict()["sampled_at"] == "2026-07-14T18:30:00+00:00"


def test_unavailable_probes_stay_none_and_report_diagnostics(monkeypatch, tmp_path) -> None:
    def unavailable(*args, **kwargs):
        raise OSError("probe denied")

    monkeypatch.setattr(resources.shutil, "disk_usage", unavailable)
    capacity = sample_workstation_capacity(
        tmp_path,
        pid=444,
        memory_probe=lambda: None,
        cpu_probe=unavailable,
        process_probe=lambda pid: None,
        accelerator_probe=unavailable,
        now=NOW,
    )

    assert capacity.disk is None
    assert capacity.memory is None
    assert capacity.cpu_percent is None
    assert capacity.process is None
    assert capacity.accelerator is None
    assert any(error.startswith("disk:") for error in capacity.errors)
    assert "memory: unavailable" in capacity.errors
    assert "process: unavailable" in capacity.errors
    assert any(error.startswith("accelerator:") for error in capacity.errors)


def test_capacity_sampling_can_skip_accelerator_probe(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        resources.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=10, used=2, free=8),
    )
    called = False

    def accelerator_probe():
        nonlocal called
        called = True
        return AcceleratorCapacity(backend="cuda")

    capacity = sample_workstation_capacity(
        tmp_path,
        memory_probe=lambda: MemoryCapacity(10, 2, 8),
        cpu_probe=lambda: None,
        accelerator_probe=accelerator_probe,
        include_accelerator=False,
        now=NOW,
    )
    assert capacity.accelerator is None
    assert not called


def test_preflight_applies_disk_and_ram_reserves() -> None:
    passed = evaluate_capacity_preflight(
        _capacity(),
        projected_disk_bytes=10 * GIB,
        projected_ram_bytes=4 * GIB,
    )
    assert passed.allowed and passed.capacity_sufficient
    assert passed.required_disk_reserve_bytes == 20 * GIB
    assert passed.required_ram_reserve_bytes == pytest.approx(3.2 * GIB)
    assert passed.projected_disk_free_bytes == 30 * GIB
    assert passed.projected_ram_available_bytes == 12 * GIB

    disk_blocked = evaluate_capacity_preflight(
        _capacity(), projected_disk_bytes=25 * GIB, projected_ram_bytes=4 * GIB
    )
    assert not disk_blocked.allowed
    assert disk_blocked.blockers == ("insufficient_disk",)

    ram_blocked = evaluate_capacity_preflight(
        _capacity(), projected_disk_bytes=10 * GIB, projected_ram_bytes=14 * GIB
    )
    assert not ram_blocked.allowed
    assert ram_blocked.blockers == ("insufficient_ram",)


def test_preflight_override_requires_and_records_nonblank_reason() -> None:
    blocked = evaluate_capacity_preflight(
        _capacity(disk_free=10 * GIB),
        override_reason="   ",
    )
    assert not blocked.allowed
    assert not blocked.overridden
    assert blocked.override_reason is None

    overridden = evaluate_capacity_preflight(
        _capacity(disk_free=10 * GIB),
        override_reason="Temporary scratch volume will be cleaned after export.",
    )
    assert overridden.allowed
    assert not overridden.capacity_sufficient
    assert overridden.overridden
    assert overridden.override_reason.startswith("Temporary scratch")
    assert overridden.blockers == ("insufficient_disk",)


def test_preflight_treats_unmeasured_capacity_as_blocking() -> None:
    unknown = WorkstationCapacity(NOW, disk=None, memory=None)
    result = evaluate_capacity_preflight(unknown)
    assert not result.allowed
    assert result.blockers == ("disk_capacity_unavailable", "ram_capacity_unavailable")
    assert result.current_disk_free_bytes is None
    assert result.current_ram_available_bytes is None

    missing_available = WorkstationCapacity(
        NOW,
        disk=DiskCapacity("/", 100 * GIB, 60 * GIB, 40 * GIB),
        memory=MemoryCapacity(32 * GIB, None, None),
    )
    assert evaluate_capacity_preflight(missing_available).blockers == ("ram_capacity_unavailable",)


def test_preflight_policy_and_inputs_are_validated() -> None:
    with pytest.raises(TypeError, match="integer"):
        evaluate_capacity_preflight(_capacity(), projected_disk_bytes=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least 0"):
        evaluate_capacity_preflight(_capacity(), projected_ram_bytes=-1)
    with pytest.raises(ValueError, match="not exceed 1"):
        CapacityPreflightPolicy(ram_reserve_fraction=1.1)


def test_two_second_telemetry_shape_preserves_unavailable_values() -> None:
    capacity = _capacity()
    sample = WorkstationTelemetrySample.from_capacity(
        "work-1",
        capacity,
        attempt_id="attempt-1",
        throughput_tokens_per_second=42.5,
        metadata={"backend": "mlx"},
    )
    payload = sample.to_dict()
    assert payload["interval_seconds"] == 2.0
    assert payload["process_rss_bytes"] == 2 * GIB
    assert payload["gpu_percent"] is None
    assert payload["power_watts"] is None
    assert payload["throughput_tokens_per_second"] == 42.5
    assert payload["metadata"]["backend"] == "mlx"
    assert payload["metadata"]["accelerator_backend"] == "mlx"
    assert payload["metadata"]["memory_available_is_estimate"] is False
    with pytest.raises(TypeError):
        sample.metadata["new"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="2 seconds"):
        WorkstationTelemetrySample("work", None, NOW, interval_seconds=1)


def test_telemetry_rollups_track_coverage_without_filling_gaps() -> None:
    samples = (
        WorkstationTelemetrySample(
            "work-1",
            "attempt-1",
            NOW,
            cpu_percent=10,
            process_rss_bytes=100,
            gpu_percent=None,
            temperature_c=-5,
        ),
        WorkstationTelemetrySample(
            "work-1",
            "attempt-1",
            NOW + timedelta(seconds=2),
            cpu_percent=20,
            process_rss_bytes=200,
            gpu_percent=None,
            temperature_c=5,
        ),
        WorkstationTelemetrySample(
            "work-1",
            "attempt-1",
            NOW + timedelta(seconds=4),
            cpu_percent=None,
            process_rss_bytes=300,
            gpu_percent=50,
            temperature_c=None,
        ),
    )
    rollup = aggregate_telemetry(samples)
    assert rollup.sample_count == 3
    assert rollup.attempt_id == "attempt-1"
    assert rollup.metric("cpu_percent").count == 2
    assert rollup.metric("cpu_percent").mean == 15
    assert rollup.metric("process_rss_bytes").maximum == 300
    assert rollup.metric("gpu_percent").count == 1
    assert rollup.metric("gpu_percent").mean == 50
    assert rollup.metric("power_watts").count == 0
    assert rollup.metric("power_watts").mean is None
    assert rollup.metric("temperature_c").minimum == -5


def test_telemetry_aggregation_partitions_attempts_and_rejects_mixed_work() -> None:
    samples = (
        WorkstationTelemetrySample("work-1", "attempt-b", NOW, cpu_percent=2),
        WorkstationTelemetrySample("work-1", "attempt-a", NOW, cpu_percent=1),
    )
    rollups = aggregate_telemetry_by_attempt(samples)
    assert [rollup.attempt_id for rollup in rollups] == ["attempt-a", "attempt-b"]
    assert aggregate_telemetry(samples).attempt_id is None

    with pytest.raises(ValueError, match="mix work items"):
        aggregate_telemetry(
            (
                samples[0],
                WorkstationTelemetrySample("work-2", "attempt-a", NOW),
            )
        )
    with pytest.raises(ValueError, match="at least one"):
        aggregate_telemetry(())


def test_raw_telemetry_retention_cutoff_is_exactly_thirty_days() -> None:
    cutoff = raw_telemetry_retention_cutoff(now=NOW)
    assert cutoff == NOW - timedelta(days=30)
    assert cutoff.tzinfo == timezone.utc
    with pytest.raises(ValueError, match="timezone-aware"):
        raw_telemetry_retention_cutoff(now=datetime(2026, 7, 14))
    with pytest.raises(ValueError, match="at least 1"):
        raw_telemetry_retention_cutoff(now=NOW, retention_days=0)


def test_linux_memory_fallback_uses_memavailable_without_optional_dependency(tmp_path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       1000 kB\nMemFree:         100 kB\nMemAvailable:    400 kB\n",
        encoding="utf-8",
    )
    memory = resources._linux_memory_probe(meminfo)
    assert memory.total_bytes == 1000 * 1024
    assert memory.available_bytes == 400 * 1024
    assert memory.used_bytes == 600 * 1024
    assert memory.source == "proc_meminfo"
    assert not memory.available_is_estimate


def test_macos_memory_fallback_marks_reclaimable_pages_as_estimated(monkeypatch) -> None:
    def run(command, **kwargs):
        if command[0] == "sysctl":
            return SimpleNamespace(returncode=0, stdout="1048576\n")
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "Mach Virtual Memory Statistics: (page size of 4096 bytes)\n"
                "Pages free:                               10.\n"
                "Pages inactive:                           20.\n"
                "Pages speculative:                         5.\n"
            ),
        )

    monkeypatch.setattr(resources.subprocess, "run", run)
    memory = resources._darwin_memory_probe()
    assert memory.total_bytes == 1_048_576
    assert memory.available_bytes == 35 * 4096
    assert memory.used_bytes == 1_048_576 - 35 * 4096
    assert memory.source == "darwin_vm_stat"
    assert memory.available_is_estimate

    capacity = WorkstationCapacity(NOW, disk=_capacity().disk, memory=memory)
    assert evaluate_capacity_preflight(
        capacity,
        policy=CapacityPreflightPolicy(
            minimum_disk_reserve_bytes=0,
            minimum_ram_reserve_bytes=0,
            disk_reserve_fraction=0,
            ram_reserve_fraction=0,
        ),
    ).ram_available_is_estimate
