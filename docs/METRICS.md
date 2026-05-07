# Prometheus metrics

Halo-forge exposes a single `GET /metrics` endpoint in Prometheus exposition format. Built from the existing telemetry providers + run database — no new collection infrastructure, just a different rendering of state we already track.

## Quick start

```bash
# Start halo-forge serving the public API.
halo-forge serve --model X --host 0.0.0.0 --port 8000  # or any other port

# Scrape the metrics:
curl http://127.0.0.1:8000/metrics
```

A scrape from a remote Prometheus needs a bearer token (the `/metrics` path honors the same loopback bypass + token gate as the rest of the public API). Local sidecar scrapes are zero-config.

## Metric reference

| Metric | Type | Labels | What it reports |
|---|---|---|---|
| `halo_forge_gpu_utilization_percent` | gauge | backend, device | Accelerator utilization 0-100 |
| `halo_forge_vram_used_gigabytes` | gauge | backend, device | VRAM in use, GB |
| `halo_forge_vram_total_gigabytes` | gauge | backend, device | VRAM total, GB |
| `halo_forge_power_watts` | gauge | backend, device | Instantaneous power draw |
| `halo_forge_temperature_celsius` | gauge | backend, device | Accelerator temperature |
| `halo_forge_cpu_utilization_percent` | gauge | backend, device | Host CPU utilization |
| `halo_forge_system_memory_used_gigabytes` | gauge | backend, device | Host RAM in use |
| `halo_forge_throughput_tokens_per_second` | gauge | backend, device | Aggregate generation throughput across active runs |
| `halo_forge_runs_total` | gauge | — | Runs indexed in the run database |
| `halo_forge_active_runs` | gauge | — | Currently running training jobs |
| `halo_forge_runs_by_modality` | gauge | modality | Runs grouped by training modality |
| `halo_forge_runs_by_status` | gauge | status | Runs grouped by completion status |
| `halo_forge_build_info` | gauge | backend, device | Always 1; labels carry backend identity |

Missing telemetry values render as `NaN` so you can alert on "metric stopped reporting" instead of guessing whether the value is zero or absent.

## Grafana dashboard

A starter dashboard JSON ships at [`docs/grafana-dashboard.json`](grafana-dashboard.json). Import it through the Grafana UI (Dashboards → Import → Upload JSON) and point it at your Prometheus data source.

The dashboard covers:

- **Top row**: backend identity, total runs, active runs, current power draw.
- **Middle**: GPU utilization, VRAM used vs total, power, temperature, CPU.
- **Bottom**: runs grouped by modality and by status.

## Prometheus scrape config

```yaml
scrape_configs:
  - job_name: halo-forge
    static_configs:
      - targets: ['halo-forge.local:8000']
    metrics_path: /metrics
    scrape_interval: 15s
    # Required when scraping a non-loopback halo-forge:
    authorization:
      type: Bearer
      credentials_file: /var/run/secrets/halo-forge-token
```

Generate the token with `halo-forge token create prometheus` and write the secret to `credentials_file`. See [`docs/auth/`](auth/) for the auth model.

## Cardinality

Halo-forge's metrics are intentionally low-cardinality:

- `backend` and `device` are bounded by the active host's accelerator (one combination per process).
- `modality` is bounded by the small set of trainer types (sft / dpo / grpo / raft / rm / vlm / audio / reasoning / agentic).
- `status` is bounded by the small set of run states.

So a single halo-forge process emits ~25-30 series total. No risk of label explosion.
