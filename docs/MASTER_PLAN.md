# Master Plan Reference

Last updated: 2026-02-11

## Purpose

This document is the project reference plan for stabilizing and hardening `halo-forge` after a deep codebase review.
It consolidates:

- Confirmed high-risk findings
- Prioritized remediation phases
- Per-component action plans
- Validation and test gates

## Scope

The review covered:

- Core CLI and orchestration (`halo_forge/cli.py`)
- Benchmarking stack (`halo_forge/benchmark/*`)
- RLVR verifiers and RAFT trainer (`halo_forge/rlvr/*`)
- SFT dataset and training flow (`halo_forge/sft/*`)
- Inference optimization/export (`halo_forge/inference/*`)
- VLM/Audio/Reasoning/Agentic modules
- UI pages/services/state/event bus (`ui/*`)
- Packaging, dependency declarations, and test posture

## Severity Model

- `P0`: correctness or contract-breaker; blocks trusted usage
- `P1`: security/runtime hardening required for safe operation
- `P2`: consistency, reliability, and maintainability
- `P3`: roadmap and release engineering improvements

## Confirmed Findings

### P0 Critical Correctness

1. `benchmark eval` Python benchmark semantics are wrong.
   - `humaneval` / `mbpp` / `livecodebench` are routed through native prompt runner paths that are not dataset-faithful.
   - Files: `halo_forge/benchmark/__init__.py`, `halo_forge/benchmark/runner.py`, `halo_forge/cli.py`.

2. UI benchmark output contract mismatch (directory-like path passed where file path is expected).
   - Produces non-standard result artifacts and breaks discoverability.
   - Files: `ui/pages/benchmark.py`, `ui/services/benchmark_service.py`, `halo_forge/cli.py`, `ui/services/results_service.py`, `ui/pages/results.py`.

3. `raft train --verifier` accepts options not handled by runtime dispatch.
   - Parser includes unsupported values in one flow, then runtime errors as unknown verifier.
   - File: `halo_forge/cli.py`.

4. Hardcoded `python` executable in runtime services.
   - Environment may only provide `python3` / `sys.executable`.
   - Files: `ui/services/training_service.py`, `ui/services/benchmark_service.py`, `halo_forge/cli.py`, verifier helpers.

5. QAT baseline integrity issue in inference optimizer.
   - Baseline and optimized model references can alias the same mutable object.
   - Files: `halo_forge/inference/optimizer.py`, `halo_forge/inference/quantization.py`.

### P1 Security and Hardening

6. Command injection risk in custom subprocess verifier (`shell=True` with template substitution).
   - File: `halo_forge/rlvr/verifiers/custom.py`.

7. `ExecutionVerifier` runs untrusted binaries without the RLIMIT protections used in compile verifier path.
   - Files: `halo_forge/rlvr/verifiers/execution.py`, `halo_forge/rlvr/verifiers/compile.py`.

8. MinGW constructor mismatch in benchmark factory.
   - Unsupported `run_after_compile` arg passed to `MinGWVerifier`.
   - Files: `halo_forge/benchmark/__init__.py`, `halo_forge/rlvr/verifiers/compile.py`.

9. Multi-language verifier option passthrough bug.
   - `hasattr` checks class attributes instead of constructor args, so flags are silently dropped.
   - File: `halo_forge/rlvr/verifiers/multi_language.py`.

10. UI HTML rendering with `sanitize=False` on dynamic content (XSS risk).
   - Files: `ui/pages/datasets.py`, `ui/pages/verifiers.py`.

### P2 Reliability and Consistency

11. UI SFT dataset options diverge from core SFT dataset registry.
   - Selections can fail at runtime.
   - Files: `ui/pages/training.py`, `halo_forge/sft/datasets.py`.

12. Benchmark preset naming mismatches in modality registries.
   - Examples: `commonvoice` vs `common_voice`, `mmstar` missing in current VLM dataset registry.
   - Files: `ui/services/benchmark_service.py`, `halo_forge/audio/data/loaders.py`, `halo_forge/vlm/data/loaders.py`.

13. VLM verifier returns `details` as dict while base verifier type expects string.
   - Can break base helpers such as `VerifyResult.__repr__`.
   - Files: `halo_forge/vlm/verifiers/base.py`, `halo_forge/rlvr/verifiers/base.py`.

14. OCR verifier hard-forces GPU usage.
   - File: `halo_forge/vlm/verifiers/perception.py`.

15. Stop-state race in services.
   - `stop_job()` marks stopped, stream-completion path can overwrite status afterward.
   - Files: `ui/services/training_service.py`, `ui/services/benchmark_service.py`.

16. GPU summary drops valid zero values due truthiness checks.
   - File: `ui/services/hardware.py`.

17. Inference verifier lacks empty-latency guards.
   - File: `halo_forge/inference/verifier.py`.

18. `quantize_model_simple` uses `tqdm` without guaranteed import in that function scope.
   - File: `halo_forge/inference/quantization.py`.

19. Dashboard “Run Benchmark” quick action points to training page query instead of benchmark route.
   - Files: `ui/pages/dashboard.py`, `ui/app.py`.

20. Results ingestion is duplicated and inconsistent.
   - Separate parsing logic in `ui/services/results_service.py` and `ui/pages/results.py`.

### P3 Product and Release Engineering

21. Modality RAFT trainers are unevenly implemented.
   - VLM/audio/agentic/reasoning contain placeholder or non-training sections in key loops.
   - Files: `halo_forge/vlm/trainer.py`, `halo_forge/audio/trainer.py`, `halo_forge/agentic/trainer.py`, `halo_forge/reasoning/trainer.py`.

22. Dependency declaration drift between `pyproject.toml` and `requirements.txt`.

23. No CI workflows detected in `.github/workflows`.

24. `halo_forge/tui` source appears incomplete (cache artifacts present, source modules absent).

## Master Remediation Plan

## Phase 0: Correctness Blockers (P0)

Goal: make core training/benchmark paths trustworthy.

Actions:

1. Fix Python benchmark routing in `run_benchmark` native path.
2. Normalize benchmark output contract to explicit result file paths.
3. Align CLI parser choices with handler support for RAFT verifier values.
4. Replace hardcoded `python` invocations with `sys.executable`.
5. Fix QAT baseline handling so baseline and candidate are distinct.

Acceptance criteria:

1. `benchmark eval --benchmark humaneval|mbpp|livecodebench` uses benchmark-appropriate dataset flow.
2. UI benchmark run produces valid JSON result files discoverable by results page and service.
3. Every parser-accepted verifier value is either fully supported or rejected at parse-time.
4. UI and CLI subprocess launches work on systems with only `python3`.
5. QAT quality verification compares against true pre-QAT baseline.

## Phase 1: Security and Runtime Hardening (P1)

Goal: reduce exploitability and unsafe execution behavior.

Actions:

1. Refactor `SubprocessVerifier` to no-shell execution model.
2. Apply resource limits to execution-based verifier path.
3. Fix MinGW mismatch and multi-language verifier argument propagation.
4. Remove `sanitize=False` usages or strictly escape untrusted content.

Acceptance criteria:

1. No verifier path executes user-influenced command text with shell expansion.
2. Untrusted binaries run under explicit CPU/memory/process limits.
3. Multi-language options (`run_after_compile`, binary cache) propagate as intended.
4. Dynamic UI content cannot inject executable HTML/JS.

## Phase 2: Consistency and Contract Cleanup (P2)

Goal: remove silent failure surfaces and schema drift.

Actions:

1. Unify UI dataset/preset keys with core registries.
2. Standardize verifier result schema (`details` typing and metadata usage).
3. Add CPU fallback behavior for OCR verifier initialization.
4. Fix service stop/completion race and enforce terminal-state idempotency.
5. Correct zero-value rendering behavior in hardware summary.
6. Add empty-input guards in inference verification.

Acceptance criteria:

1. Any UI selectable dataset/preset is runnable end-to-end.
2. `VerifyResult` contract is consistent across all verifier families.
3. Jobs cannot transition from terminal state to conflicting terminal state.
4. Hardware UI correctly shows `0%` utilization and `0.0` memory when real.

## Phase 3: Modality Pipeline Truth-in-Advertising (P3)

Goal: align claims, UX, and implementation depth.

Actions:

1. For VLM/audio/reasoning/agentic trainers: either implement true fine-tuning loops or mark as evaluation/prototype with explicit gating.
2. Ensure CLI help and docs reflect actual capability status.

Acceptance criteria:

1. No command implies production training if it does not train model weights.
2. Modality training status is explicit and test-backed.

## Phase 4: UI and Observability Consolidation (P2/P3)

Goal: simplify operational reliability.

Actions:

1. Consolidate results parsing into one canonical service used by all pages.
2. Correct dashboard route wiring for benchmark quick actions.
3. Improve event/state traces for job lifecycle transitions.

Acceptance criteria:

1. Single source of truth for result ingestion.
2. Dashboard action links match actual routes and page behavior.
3. Lifecycle telemetry supports diagnosis of stuck/failed/stopped states.

## Phase 5: Release Engineering and Regression Net (P3)

Goal: prevent reintroduction of known classes of defects.

Actions:

1. Add CI workflows (`lint`, `unit`, selective integration smoke checks).
2. Reconcile dependency strategy between `pyproject.toml` and `requirements.txt`.
3. Add regression tests for each fixed P0/P1 issue.
4. Audit `halo_forge/tui` status and restore or remove stale packaging hooks.

Acceptance criteria:

1. CI required checks pass on PRs.
2. Dependency installation paths are deterministic and documented.
3. Regression tests cover benchmark routing, output contract, verifier dispatch, and execution hardening.

## Component-Level Worklists

### CLI / Orchestration

- Split parser definitions from runtime handlers to prevent drift.
- Add shared verifier registry for CLI/UI/service consumers.
- Replace ad-hoc subprocess command builders with validated command factories.

### Benchmarking

- Separate “standard benchmark datasets” from “internal prompt suites” explicitly.
- Keep `pass_at_k` path and `BenchmarkRunner` path well-scoped and non-overlapping.
- Enforce output schema and file naming convention for all benchmark commands.

### RLVR Verifiers

- Standardize all verifier constructors and option propagation.
- Harden subprocess and execution boundaries.
- Document per-verifier safety posture.

### SFT and Data

- Use central dataset registry as UI source of truth.
- Add schema validation for local datasets before launch.

### Inference

- Protect baseline/candidate separation.
- Ensure all verification metrics are safe on edge inputs.
- Fix quantization utility imports and dry-run diagnostics.

### UI / Services / State

- Centralize job state transitions and terminal-state guards.
- Remove duplicated result parsers.
- Route and preset normalization against backend registries.

## Test Plan

Add/expand tests for:

1. Benchmark routing correctness by benchmark name.
2. Benchmark output path behavior from UI service to CLI.
3. RAFT verifier parser-handler parity.
4. `sys.executable` subprocess portability in services.
5. QAT baseline integrity verification.
6. Custom verifier command execution hardening.
7. Execution verifier resource limiting.
8. UI dataset/preset key compatibility.
9. Job stop/completion race handling.
10. Hardware zero-value formatting.

## Rollout Order

1. Phase 0 + critical tests.
2. Phase 1 + security review.
3. Phase 2 + compatibility cleanup.
4. Phase 3 capability alignment.
5. Phase 4 observability consolidation.
6. Phase 5 CI and release hardening.

## Working Notes

- Syntax verification previously passed with `python3 -m compileall -q halo_forge ui`.
- Runtime validation requires project dependencies (`pytest`, `torch`, modality deps) in active environment.

