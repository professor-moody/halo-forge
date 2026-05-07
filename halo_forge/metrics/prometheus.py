"""Prometheus exposition format helpers."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


# ---------- low-level formatter --------------------------------------------


def _escape_label(value: str) -> str:
    """Escape a label value per Prometheus exposition format."""
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )


def _format_metric_line(
    name: str,
    value: float,
    labels: Optional[Dict[str, Any]] = None,
) -> str:
    """One sample line: ``name{k="v",k="v"} 0.42``."""
    if labels:
        rendered_labels = ",".join(
            f'{k}="{_escape_label(v)}"'
            for k, v in sorted(labels.items())
            if v is not None
        )
        return f"{name}{{{rendered_labels}}} {value}"
    return f"{name} {value}"


def format_metrics(
    *,
    metrics: List[Dict[str, Any]],
) -> str:
    """Render ``metrics`` as Prometheus exposition text.

    Each entry is::

        {
            "name": "halo_forge_gpu_utilization_percent",
            "help": "GPU utilization (0-100)",
            "type": "gauge",      # gauge | counter | histogram | summary
            "samples": [
                {"value": 42.0, "labels": {"backend": "cuda"}},
                ...
            ],
        }

    Empty metrics still emit the HELP / TYPE preamble so a scraper
    can tell "no samples now" from "metric doesn't exist".
    """
    lines: List[str] = []
    for metric in metrics:
        name = str(metric["name"])
        help_text = str(metric.get("help", "")).replace("\n", " ")
        m_type = str(metric.get("type", "gauge"))
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {m_type}")
        for sample in metric.get("samples") or []:
            value = sample.get("value")
            if value is None:
                # Prometheus convention: emit NaN for missing values so
                # the metric line still appears (helpful for
                # alerting on "metric stopped reporting").
                value = "NaN"
            lines.append(
                _format_metric_line(
                    name,
                    value,
                    labels=sample.get("labels") or None,
                )
            )
    return "\n".join(lines) + "\n"


# ---------- halo-forge specific renderer -----------------------------------


def render_metrics(
    *,
    telemetry: Optional[Dict[str, Any]] = None,
    run_stats: Optional[Dict[str, Any]] = None,
    backend_info: Optional[Dict[str, Any]] = None,
) -> str:
    """Render halo-forge's full metric surface to Prometheus text.

    Args:
        telemetry: A `TelemetrySample.to_dict()` result. None → metrics
            still emit HELP/TYPE but no samples.
        run_stats: ``{
            "total_runs": int,
            "by_modality": {"sft": 12, "raft": 5, ...},
            "by_status": {"completed": 8, "failed": 2, ...},
            "active_runs": int,
        }`` — typically pulled from `RunDatabase` + the live job table.
        backend_info: ``halo_forge.backend.get_backend()`` result for
            backend-name + dtype labels on the gauges.
    """
    backend = (telemetry or {}).get("backend") or (backend_info or {}).get("name") or "unknown"
    device = (telemetry or {}).get("device_name") or (backend_info or {}).get("device") or "unknown"
    base_labels = {"backend": backend, "device": device}

    metrics: List[Dict[str, Any]] = []

    # --- GPU + system telemetry (gauges from the live sample) ---

    metrics.append({
        "name": "halo_forge_gpu_utilization_percent",
        "help": "GPU utilization percent (0-100). NaN when unavailable.",
        "type": "gauge",
        "samples": [
            {
                "value": _opt_float((telemetry or {}).get("gpu_util_percent")),
                "labels": base_labels,
            },
        ],
    })
    metrics.append({
        "name": "halo_forge_vram_used_gigabytes",
        "help": "Accelerator VRAM in use, gigabytes.",
        "type": "gauge",
        "samples": [{
            "value": _opt_float((telemetry or {}).get("vram_used_gb")),
            "labels": base_labels,
        }],
    })
    metrics.append({
        "name": "halo_forge_vram_total_gigabytes",
        "help": "Accelerator VRAM total, gigabytes.",
        "type": "gauge",
        "samples": [{
            "value": _opt_float((telemetry or {}).get("vram_total_gb")),
            "labels": base_labels,
        }],
    })
    metrics.append({
        "name": "halo_forge_power_watts",
        "help": "Accelerator instantaneous power draw, watts.",
        "type": "gauge",
        "samples": [{
            "value": _opt_float((telemetry or {}).get("power_watts")),
            "labels": base_labels,
        }],
    })
    metrics.append({
        "name": "halo_forge_temperature_celsius",
        "help": "Accelerator temperature, °C.",
        "type": "gauge",
        "samples": [{
            "value": _opt_float((telemetry or {}).get("temp_celsius")),
            "labels": base_labels,
        }],
    })
    metrics.append({
        "name": "halo_forge_cpu_utilization_percent",
        "help": "Host CPU utilization percent (0-100).",
        "type": "gauge",
        "samples": [{
            "value": _opt_float((telemetry or {}).get("cpu_util_percent")),
            "labels": base_labels,
        }],
    })
    metrics.append({
        "name": "halo_forge_system_memory_used_gigabytes",
        "help": "Host RAM in use, gigabytes.",
        "type": "gauge",
        "samples": [{
            "value": _opt_float((telemetry or {}).get("sys_mem_used_gb")),
            "labels": base_labels,
        }],
    })
    metrics.append({
        "name": "halo_forge_throughput_tokens_per_second",
        "help": "Aggregate generation throughput across active runs.",
        "type": "gauge",
        "samples": [{
            "value": _opt_float((telemetry or {}).get("throughput_tokens_per_sec")),
            "labels": base_labels,
        }],
    })

    # --- Run inventory (gauges from the run DB + live jobs) ---

    stats = run_stats or {}
    total = int(stats.get("total_runs") or 0)
    metrics.append({
        "name": "halo_forge_runs_total",
        "help": "Total runs indexed in the run database.",
        "type": "gauge",
        "samples": [{"value": float(total), "labels": None}],
    })
    metrics.append({
        "name": "halo_forge_active_runs",
        "help": "Currently running training jobs (in the live job table).",
        "type": "gauge",
        "samples": [{
            "value": float(stats.get("active_runs") or 0),
            "labels": None,
        }],
    })

    # Per-modality + per-status counts surface as labeled samples on
    # one metric each. Empty dicts emit the HELP/TYPE only.
    by_modality = (stats.get("by_modality") or {})
    metrics.append({
        "name": "halo_forge_runs_by_modality",
        "help": "Runs grouped by training modality.",
        "type": "gauge",
        "samples": [
            {"value": float(count), "labels": {"modality": modality}}
            for modality, count in sorted(by_modality.items())
        ],
    })
    by_status = (stats.get("by_status") or {})
    metrics.append({
        "name": "halo_forge_runs_by_status",
        "help": "Runs grouped by completion status.",
        "type": "gauge",
        "samples": [
            {"value": float(count), "labels": {"status": status}}
            for status, count in sorted(by_status.items())
        ],
    })

    # --- Build / version label ---

    metrics.append({
        "name": "halo_forge_build_info",
        "help": "Build identity (always 1; labels carry the metadata).",
        "type": "gauge",
        "samples": [{
            "value": 1.0,
            "labels": {
                "backend": backend,
                "device": device,
            },
        }],
    })

    return format_metrics(metrics=metrics)


def _opt_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["format_metrics", "render_metrics"]
