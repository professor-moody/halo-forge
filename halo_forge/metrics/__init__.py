"""Prometheus exposition (Track P3).

A single ``GET /metrics`` endpoint, plain-text Prometheus format. Built
from the existing telemetry providers + run DB stats — no new
collection infrastructure, just a different rendering of state we
already track.

Why a halo-forge-shaped Prometheus exporter rather than letting users
wire their own ``prometheus_client``: most halo-forge users deploy
the public API as a single process; integrating ``prometheus_client``
would force a second port + lifecycle. The metrics here render in
~1ms from cached state and stream into any Prometheus scrape with
zero additional plumbing.
"""

from halo_forge.metrics.prometheus import format_metrics, render_metrics

__all__ = ["format_metrics", "render_metrics"]
