# Ops Dataset Burn-In Summary

This summary tracks the public, sanitized runtime-contract state for Phase 7I non-code burn-in.

## Scope
- Modules: `vlm`, `audio`, `reasoning`, `agentic`, `inference`, `benchmark`, `ui_ops`
- Profile: `tiny-v1`
- Seed: `42`
- Policy: PR/push informational, nightly strict

## Canonical Artifacts
- Report contract path: `results/readiness/ops_dataset_burnin.v1.json`
- Baseline snapshot: `tests/baselines/ops_dataset_burnin_baseline.v1.json`
- Runner: `scripts/run_ops_dataset_burnin.py`

## Status Semantics
- `pass`: required contract checks and required artifacts are valid
- `warn`: required contracts pass, but optional/runtime warnings are present
- `fail`: required contracts or required artifacts are invalid

## CI Enforcement
- PR/push: non-strict burn-in report generation for visibility
- Nightly: strict burn-in + baseline compare; fails on:
  - module `status=fail`
  - hard contract drift in baseline comparison

## Notes
- This summary intentionally excludes internal-only planning details and private evidence.
- For internal triage packets, use `.internal_docs/research_testing/packets/`.
