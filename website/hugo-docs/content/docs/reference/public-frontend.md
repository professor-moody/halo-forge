---
title: "Public Frontend"
description: "Public-facing training, monitor, results, and readiness surface"
weight: 6
---

The public halo-forge product surface is now split from the internal NiceGUI console.

## Public Surface

The public frontend is a dedicated Next.js app aimed at normal users:

- **Train**: launch training with quickstart presets and a plain-language quality outlook
- **Runs**: watch live progress with one recommended next step and expandable research detail
- **Results**: review outcomes, recovery guidance, and follow-up actions
- **Readiness**: see modality qualification status without opening the internal console
- **Docs**: consume repo-truth capability summaries and workflow guidance

The public frontend reads from the standalone public API:

```bash
uvicorn halo_forge.public_api.app:app --host 127.0.0.1 --port 8081
```

Set `NEXT_PUBLIC_HALO_API_BASE` when running the frontend if the API is not on `http://127.0.0.1:8081/api/public`.

## Internal Console

halo-forge ui remains the internal ops/research console.

Use the NiceGUI console for:

- raw logs and deep trace inspection
- advanced diagnostics tools
- ops workflows and setup remediation
- internal-only debugging surfaces

Use the public frontend for the default product workflow.

## Product Rules

- One primary action per state
- Plain-language status first
- Research details available but collapsed by default
- Internal reason codes and low-level traces are not shown in the primary UI
- Readiness and capability copy should follow qualification truth, not hand-written claims
