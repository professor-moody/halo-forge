# Ops Readiness Fixture Packs

Fixture packs under this directory provide deterministic contract-only inputs for strict
ops-readiness validation in nightly CI.

## v1

Path: `tests/fixtures/ops_readiness/v1`

Modules covered:
- `vlm`
- `audio`
- `reasoning`
- `agentic`
- `inference`
- `benchmark`
- `ui_ops`

Notes:
- `ui_ops` strict validation uses repository root wiring checks; the fixture pack includes a
  placeholder `ui_ops/` directory for contract completeness.
- Run strict validation with:

```bash
python3 scripts/run_ops_module_matrix.py --fixture-pack v1 --strict
```
