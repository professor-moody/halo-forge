Deterministic fixture pack for ops E2E launch reliability checks.

`v1/` mirrors non-code module output contracts used by:
- `scripts/run_ops_e2e_reliability.py --fixture-pack v1`
- `halo-forge test --level ops-e2e --fixture-pack v1`

Each module directory contains the minimum lifecycle artifacts required by the
E2E contract validator (launch context, terminal artifacts, and relaunch hooks).
