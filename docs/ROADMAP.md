# Halo Forge Product and Engineering Roadmap

- **Updated:** 2026-08-23
- **Current release line:** 2.0.0-alpha-2
- **Planning horizon:** alpha-2 closure through 2.0 general availability

Halo Forge has enough product breadth. The road to 2.0 is now a reliability,
maintainability, security, qualification, and usability program rather than a
feature-count program.

This roadmap is gate-driven. Timeboxes are planning estimates, not promises. A
milestone moves when its exit criteria are met; an arbitrary date does not turn
missing evidence into support.

## Product objective

Halo Forge should let an operator take owned data to a locally controlled,
evaluated, reproducible, and serveable model artifact through one understandable
workflow:

```text
Own data
  -> inspect and publish an immutable dataset
  -> receive one explainable training recommendation
  -> prove that the path fits and updates weights
  -> compare compatible evaluation evidence
  -> review the outcome
  -> promote, serve, or export the qualified artifact
```

Advanced experiment, verifier, reward-integrity, review, adaptation-study, and
environment workflows remain first-class, but they must not obscure this
primary path.

## Roadmap principles

1. **Truth before breadth.** A capability is advertised only when the exact
   path has current evidence on the named runtime and hardware family.
2. **One domain model, several surfaces.** CLI, API, browser, and desktop use
   the same records and services; parity is tested rather than reimplemented.
3. **Immutable evidence.** Dataset, tokenizer, chat-template, verifier,
   runtime, run, evaluation, and artifact identity remain reconstructable.
4. **Safe local operation.** Untrusted verifier execution fails closed;
   remote binding requires authentication; desktop privileges are minimal.
5. **Reproducible releases.** Application and desktop builds use frozen,
   reviewed dependency sets while the Python package publishes deliberate
   compatibility ranges.
6. **No feature expansion during a red gate.** Release, security, migration,
   or data-integrity failures take priority over new Labs or trainer modes.
7. **Evidence expires.** Hardware and runtime qualifications carry versions,
   timestamps, and requalification triggers.

## Current baseline

The following capability layers are delivered and form the baseline to
stabilize, not a list of work to rebuild:

- Dataset Lab v1-v2: immutable multimodal datasets, recipes, trainer artifacts,
  evaluations, comparison, and failure mining;
- Lab v3-v5: experiment operations, workstation scheduling, Artifact Studio,
  adaptive checkpoints, and reproducible evidence;
- Lab v6-v8: human review, verifier reliability, reward integrity, and exact
  training-signal capture;
- Labs v11-v15: proof outcomes, controlled adaptation studies, grounded data,
  specialized task models, and deterministic environments;
- V21: exact real-path certification and workstation beta evidence;
- SFT, CPT, DPO, GRPO, RAFT, modality trainers, evaluation, conversion,
  serving, and artifact workflows across capability-checked backends; and
- React/FastAPI browser UI plus Tauri desktop packaging contracts.

The detailed contracts remain in the domain documents linked from
[Documentation](README.md). The roadmap tracks what must become dependable and
supportable next.

## Priority model

| Priority | Meaning | Examples |
|---|---|---|
| P0 | Blocks a release or can corrupt, misidentify, expose, or silently discard user work | identity errors, failed release gates, unsafe execution, broken migrations |
| P1 | Material reliability, maintainability, or primary-workflow failure | dependency drift, worker recovery, control-plane decomposition, test isolation |
| P2 | Important product quality or operational scale | accessibility, large-catalog performance, advanced workflow refinement |
| P3 | Valid future capability with no current release dependency | distributed scheduling, new search policies, remote publishing |

Every roadmap issue must name its priority, milestone, owner, acceptance test,
and user-visible effect. “Improve,” “support,” and “polish” are not sufficient
acceptance criteria by themselves.

## Critical path and parallel work

```text
Frozen dependencies and P0 correctness
  -> alpha-2 engineering evidence and developer preview
    -> control-plane decomposition and hermetic CI
      -> secure, recoverable primary workflow
        -> real-hardware qualification
          -> release candidate
            -> 2.0 GA
```

Security analysis, documentation repair, and hardware-lab scheduling may begin
earlier, but they do not bypass the chain. In particular, hardware evidence is
not release evidence until it was produced by the frozen runtime and current
identity contracts.

## Release train

### Milestone 0 — alpha-2 release closure

- **Purpose:** close the already-declared alpha-2 engineering surface and ship
  an honestly labeled developer preview without adding scope.
- **Indicative timebox:** 2-3 focused weeks.
- **Feature policy:** freeze; accept only P0/P1 release fixes and documentation
  truth corrections.
- **Status (2026-08-23):** engineering closure is complete on the release branch.
  The frozen dependency contract, full Python suite, modality/ops gates,
  frontend/docs/Cargo builds, packaged-runtime self-check, and packaged
  two-step SFT/dashboard smoke pass locally. The no-cost publication path is an
  unsigned, checksummed macOS developer preview on a GitHub prerelease; source,
  CLI, and browser installation remain the supported paths. Developer ID
  signing and notarization are a separately funded promotion gate and do not
  block moving engineering work to Milestone 1.

#### Work

- Establish a Python dependency policy:
  - commit a reviewed `uv.lock` for development and packaged runtimes;
  - use frozen resolution in CI and desktop/release builds;
  - retain intentional ranges in `pyproject.toml` for wheel consumers;
  - add release constraints for packages whose major versions are not yet
    compatible, including Transformers; and
  - test both the frozen application set and the declared package range.
- Fix all undefined names and make Ruff F82 a blocking gate. The swallowed
  Review Studio `validate_record` failure is P0 because it can reject valid
  output records without exposing the implementation error.
- Resolve the reasoning modality baseline drift for checkpoint and resume
  evidence. Update a baseline only after explaining and reviewing the behavior
  change.
- Land the chat-template identity contract with:
  - one versioned digest scheme;
  - explicit present, absent, unreadable, and unsupported states;
  - conversion and catalog round-trip coverage;
  - backward-readable legacy records; and
  - no dead or shadowed test helpers.
- Make the full Python suite pass in the frozen release environment.
- Repair or explicitly defer every non-strict release-readiness check. A
  deferred check must identify the unsupported surface in release notes.
- Regenerate and verify frontend, documentation, desktop, and packaged-runtime
  artifacts from a clean checkout.
- Publish unsigned macOS alpha/beta/RC candidates only with an
  `-unsigned-preview` filename, checksum, explicit preview manifest state, and
  opt-in installation guidance. Never attach an unsigned DMG to a stable
  release or recommend it on the normal download path.
- Keep macOS signing, notarization, and stapling conditional on credentials;
  complete those gates before calling any DMG supported.
- Reconcile version strings, release notes, changelog, download page, hardware
  matrix, and architecture documentation.

#### Exit criteria

- all blocking GitHub Actions jobs are green on the release commit;
- zero Ruff E9/F63/F7/F82 findings;
- frozen full suite passes with no unreviewed deselection or quarantine;
- modality baseline, release interface, frontend contract, Cargo, and packaged
  smoke gates pass;
- no test writes to an operator's real `~/.halo-forge` state;
- signature, notarization, availability, and support status are represented
  truthfully in the release manifest;
- unsigned macOS assets are confined to prereleases and visibly labeled as
  developer previews;
- no known P0 issue is open; and
- the release checklist can be executed from a clean clone without undocumented
  local state.

#### Funded macOS distribution gate

This gate may be completed whenever project funding or an eligible program
membership becomes available. It blocks promotion of the DMG to a supported
normal installer, but it does not block alpha engineering or the source,
browser, and CLI release paths.

- obtain a project-controlled Apple Developer membership and Developer ID
  identity without sharing personal certificates;
- store signing and notarization credentials in protected release secrets;
- validate codesign, notarization, stapling, Gatekeeper acceptance, packaged
  smoke, checksum, manifest, and public-download readback; and
- change the download page from preview guidance to a recommended DMG only
  after that exact artifact passes every gate.

### Milestone 1 — alpha-3 maintainability and deterministic CI

- **Purpose:** reduce control-plane blast radius without changing product
  semantics.
- **Indicative timebox:** 4-6 weeks.
- **Feature policy:** domain-preserving refactors only.

#### Workstream A: service decomposition

- Split `halo_forge/public_api/app.py` into domain routers: system, datasets,
  review, training, experiments, evaluations, artifacts, verifiers, rewards,
  and research Labs.
- Split `PublicApiService` into domain application services behind a temporary
  compatibility facade.
- Move CLI parser and handler registration into the same domain packages.
- Split database operations into schema/migration, run, work, dataset,
  evaluation, artifact, and research repositories while retaining one SQLite
  transaction boundary where required.
- Define and enforce dependency direction:

```text
transport (CLI/FastAPI/Tauri)
  -> application services
    -> domain services
      -> repositories and runtime adapters
```

- Add import-boundary tests so UI and transports cannot become dependencies of
  domain logic.

#### Workstream B: API and frontend contract

- Treat the FastAPI OpenAPI document as a checked artifact.
- Generate or mechanically validate the TypeScript client and shared schemas
  instead of manually mirroring hundreds of resources.
- Preserve endpoint compatibility through contract snapshots and deprecation
  periods.
- Add pagination, cancellation, idempotency, error, and asynchronous-work
  conventions shared by every domain router.

#### Workstream C: test infrastructure

- Add an autouse isolated Halo Forge state root for tests and reset cached
  databases, supervisors, token stores, and workers between cases.
- Functionally probe `sandbox-exec` or `bwrap`; skip with an explicit reason
  when the binary exists but the parent environment prevents execution.
- Separate unit, contract, integration, sandbox, packaged-runtime, hardware,
  and soak suites with documented runtimes and owners.
- Add coverage reporting and ratchet changed-code coverage before setting a
  repository-wide threshold.
- Add a supported-Python core matrix for 3.10-3.13 and retain one frozen full
  suite on the release Python version.
- Track flaky tests; quarantine requires an issue, owner, reason, and expiry.

#### Exit criteria

- no production Python module exceeds an agreed control-plane size budget
  without an architectural exception;
- adding a domain endpoint normally changes one router, one application
  service, one client surface, and its tests rather than central monoliths;
- OpenAPI/client drift fails CI;
- the full non-hardware suite is hermetic and repeatable from a clean clone;
- changed-code coverage is reported and cannot regress below the ratchet; and
- no behavior or persisted schema changes without explicit migration evidence.

### Milestone 2 — beta-1 secure, recoverable guided workstation

- **Purpose:** make the primary own-data workflow safe and understandable for
  external beta users.
- **Indicative timebox:** 4-6 weeks.

#### Security and privacy

- Publish `SECURITY.md` with supported versions, reporting instructions,
  verifier-execution boundaries, and response expectations.
- Add a maintained threat model for local browser, remote browser, desktop,
  verifier sandbox, model loading, document extraction, tokens, and support
  bundles.
- Enable a restrictive Tauri content security policy.
- Remove webview shell permissions that are not required; scope any remaining
  command and argument explicitly.
- Replace “any HTTP 200 on port 8765” desktop readiness with a product/version
  response and per-launch handshake or equivalent ownership proof.
- Verify non-loopback auth, CORS, token rotation/revocation, secret redaction,
  and support-bundle privacy in integration tests.
- Add dependency update automation, vulnerability review, SBOM generation, and
  release provenance for Python, npm, Cargo, and packaged runtimes.
- Audit archive extraction, local path handling, symlinks, subprocess
  invocation, model remote code, and verifier workspace containment.

#### Primary experience

- Make **Train on your data** the default entry point.
- Present one recommendation, its evidence, its resource estimate, and one next
  safe action at a time.
- Place raw trainer flags, JSON specifications, and research controls behind a
  clearly labeled Advanced mode.
- Unify preparation, queued work, proof progress, evaluation, and outcome
  review into a recoverable journey with durable deep links.
- Provide actionable recovery for gated models, missing runtimes, insufficient
  storage, incompatible trainers, interrupted workers, and failed artifacts.
- Ensure every irreversible or expensive action has a preview, scope, and
  confirmation.

#### Frontend quality

- Add Playwright smoke flows for setup, own-data publication, proof launch,
  run recovery, evaluation comparison, and artifact serving.
- Add automated keyboard and accessibility checks for the primary workflow.
- Split oversized frontend chunks and set explicit JavaScript budgets.
- Exercise 10,000-run/dataset/artifact catalogs for query latency, pagination,
  search, reconnect, and memory behavior.

#### Exit criteria

- no unresolved high or critical security finding in the supported surface;
- desktop capabilities and CSP pass an explicit regression test;
- all primary workflows pass browser and packaged-desktop Playwright smoke;
- a new operator can reach a qualified proof preflight without editing JSON or
  knowing trainer implementation names;
- interrupted preparation or proof work resumes or fails with a bounded,
  actionable state; and
- accessibility checks have no critical violations on primary routes.

### Milestone 3 — beta-2 real-hardware qualification

- **Purpose:** replace platform possibility with measured support evidence.
- **Indicative timebox:** 6-10 weeks, constrained by hardware access.

#### Qualification ladder

For every advertised trainer/backend pair:

1. clean install or packaged runtime;
2. runtime and accelerator identity capture;
3. immutable dataset and tokenizer preparation;
4. capacity preflight;
5. real forward/backward optimizer step;
6. trainable-parameter hash change;
7. checkpoint save and resume;
8. final artifact reload;
9. compatible base/candidate evaluation;
10. reviewed proof outcome;
11. scheduler wait/release and restart recovery; and
12. bounded soak with telemetry and integrity readback.

#### Hardware targets

- **Apple Silicon:** MLX SFT, CPT, RAFT, DPO, and GRPO on the supported MLX
  line; MPS routes only where explicitly verified.
- **AMD ROCm / Strix Halo:** managed-runtime qualification, instruction SFT
  first, then progressively certified trainer paths.
- **NVIDIA CUDA:** keep guided scenarios hardware-unqualified until the same
  ladder passes on a real NVIDIA host.
- **CPU:** metadata, validation, deterministic tests, and tiny smoke only; do
  not market it as a practical heavy-training path.

#### Operational evidence

- Run twelve-hour workstation event windows with bounded sequential proofs.
- Exercise power loss/process death, worker restart, retry, cancellation,
  external accelerator contention, low disk, and artifact publication recovery.
- Soak reward-signal capture, sealing, same-output sentinel rescoring, boundary
  review, and continuation across supported verifier-guided trainers.
- Measure rather than infer peak memory, disk growth, throughput, power, and
  energy; keep unavailable metrics null.
- Publish qualification timestamps, runtime versions, hardware identities,
  limitations, and invalidation rules in the support matrix.

#### Exit criteria

- every green support-matrix cell points to current checksummed evidence;
- no capability is enabled solely from hardware detection or a generic tensor
  test;
- the recommended Apple and AMD instruction-SFT paths complete the full ladder;
- CUDA guided scenarios remain disabled until equivalent NVIDIA evidence is
  complete;
- recovery and twelve-hour soak requirements pass on representative supported
  workstations; and
- qualification evidence is reproducible by a second operator from published
  instructions.

### Milestone 4 — 2.0 release candidate

- **Purpose:** freeze behavior and prove upgrade, compatibility, distribution,
  and support readiness.
- **Indicative timebox:** 4 weeks.
- **Feature policy:** complete freeze; P0/P1 release fixes only.

#### Work

- Freeze public CLI, API, database schema, replay, artifact manifest, and
  configuration contracts for 2.0.
- Test upgrade and rollback from the latest supported 1.x release and every
  2.0 prerelease carrying persisted-state changes.
- Verify backup, restore, cleanup, uninstall preservation, and corrupted-state
  recovery.
- Run signed/notarized macOS, signed Windows, and finalized Linux distribution
  candidates through packaged-runtime and primary-workflow smoke.
- Complete operator, troubleshooting, architecture, security, migration,
  hardware, and contributor documentation.
- Publish known limitations and distinguish preview, qualified, and unsupported
  cells consistently across code, API, UI, docs, and release metadata.
- Run a release-candidate security review and dependency/license inventory.
- Establish issue templates, ownership, triage cadence, and support boundaries.

#### Exit criteria

- zero open P0 and no unaccepted P1 release blocker;
- two consecutive clean release-candidate runs from clean checkouts;
- migrations and rollback/recovery pass against preserved representative
  catalogs;
- all distributed artifacts have checksums, provenance, signature status, and
  packaged smoke evidence;
- documentation and product claims match the qualification registry; and
- no release gate is informational unless the corresponding capability is
  explicitly excluded from 2.0 support.

### Milestone 5 — 2.0 general availability

**Purpose:** publish a supportable local model-development workstation.

#### Launch requirements

- publish signed, checksummed, manifest-qualified artifacts only;
- publish Python/source installation with reviewed constraints and a frozen
  reproducibility path;
- publish the final hardware and trainer support matrix;
- publish migration, backup, rollback, security, and troubleshooting guides;
- tag, build, verify, and read back the release from the public download and
  package surfaces; and
- open a 30-day stabilization window in which reliability and documentation
  take precedence over new capability work.

#### GA success measures

- at least 95% of qualified-path CI and scheduled hardware runs pass without
  retry over the stabilization window;
- flaky-test rate remains below 0.5%;
- zero silent data-loss, identity, migration, or security regression;
- primary local catalog queries stay below the agreed p95 budget at 10,000
  records on reference hardware;
- the largest initial frontend chunk stays within the committed bundle budget;
- support reports contain actionable identity and diagnostics without secrets,
  prompts, dataset contents, or model weights; and
- every support claim can be traced to current qualification evidence.

## Cross-cutting workstreams

These run through all milestones and should be visible as separate issue
labels or project lanes.

| Workstream | Accountable area | Typical artifacts |
|---|---|---|
| Release/platform | dependency sets, CI, packaging, migrations, release evidence | locks, constraints, workflows, manifests |
| Core architecture | service boundaries, repositories, scheduling, replay | ADRs, domain packages, migration tests |
| Product/frontend | primary journey, advanced mode, accessibility, performance | route specs, Playwright flows, bundle reports |
| Security/privacy | threat model, desktop capabilities, auth, sandbox, redaction | SECURITY.md, tests, SBOM, review reports |
| Hardware/ML | trainer truth, backend qualification, performance, soaks | certification bundles, support matrix |
| Documentation/community | architecture truth, operator guidance, contribution path | docs, examples, issue templates |

For a single-maintainer sequence, execute these lanes in milestone order. With
multiple maintainers, release/platform and security can proceed beside
architecture or product work, but no lane may bypass shared milestone gates.

## Engineering scorecard

Review this scorecard at each milestone boundary.

| Dimension | Alpha-2 gate | Beta gate | GA gate |
|---|---|---|---|
| Correctness | frozen full suite green; zero undefined names | hermetic suites and changed-code coverage ratchet | two clean RC runs; no P0/P1 blocker |
| Reproducibility | reviewed lock/constraints and frozen builds | qualification bundles reproducible | public artifacts and install paths read back |
| Architecture | documented current topology | domain routers/services and API drift gate | stable 2.0 public contracts |
| Security | known P0 fixed | threat model, CSP, minimal capabilities, SBOM | release security review complete |
| Product | existing flows truthful | primary own-data journey passes E2E | supportable onboarding and recovery |
| Hardware | no unsupported green claims | full evidence ladder on reference hosts | public current support matrix |
| Operations | core scheduler gates green | restart/retry/soak evidence | upgrade, backup, restore, uninstall proven |

## Risk register

| Risk | Consequence | Mitigation | Trigger |
|---|---|---|---|
| Product breadth outruns maintenance | regressions and confusing navigation | feature freeze, primary-path focus, domain owners | new Lab or trainer proposed before GA |
| Dependency major-version drift | clean installs fail unpredictably | frozen app set, deliberate ranges, min/latest CI | resolver changes a training dependency |
| Central modules amplify changes | unrelated domains break together | compatibility facade plus incremental decomposition | cross-domain diff for a local endpoint |
| Tests depend on user or host state | flaky failures or polluted local data | isolated roots, functional capability probes | test touches real home, port, GPU, or sandbox |
| Qualification becomes stale | support matrix overstates reality | expiry and invalidation on runtime/model changes | backend, driver, trainer, or model adapter changes |
| Desktop/web privilege boundary is too broad | local content compromise gains app privileges | CSP, minimal capability scopes, handshake | new Tauri command, remote origin, or plugin |
| Untrusted verifier execution escapes | workstation data or network exposure | fail-closed sandbox, containment tests, unsafe opt-in | sandbox/backend/profile change |
| Schema growth breaks old catalogs | loss of user history or startup failure | additive migrations, fixtures, backups, rollback tests | schema or replay version increments |

## Definition of done

A roadmap item is complete only when all applicable conditions hold:

- behavior is implemented in the domain service rather than duplicated across
  transports;
- CLI, API, browser, and desktop parity is either present or the limitation is
  explicit;
- persisted-state changes include forward migration, backward-read behavior,
  representative fixtures, and recovery guidance;
- success, failure, interruption, cancellation, retry, and idempotency paths
  are covered where relevant;
- user-visible claims are capability- and evidence-backed;
- security and privacy implications are reviewed;
- tests are deterministic and do not depend on personal workstation state;
- operator and developer documentation is updated; and
- release checks pass from a clean checkout.

## Post-2.0 candidates

These are valid directions, but none should enter the 2.0 critical path.

### Scale and scheduling

- multi-workstation orchestration and distributed training;
- additional verified pruning and multi-objective search policies;
- statistically valid sequential testing and adaptive budget allocation; and
- cross-backend normalized performance studies.

### Artifact and distribution

- verified quantization-aware training;
- reviewed publishing to Hugging Face or other registries;
- additional conversion formats with round-trip loading evidence; and
- adapter routing and continued-training policies.

### Data and research

- autonomous development-data proposals with explicit review gates;
- multi-user review assignment and agreement analysis;
- additional reproducible acquisition algorithms;
- live or nondeterministic agent environments with side-effect controls;
- binary image/audio generation; and
- broader study designs and publication workflows.

Post-2.0 work should be selected from measured user friction, support burden,
and qualification evidence—not from the availability of another organizational
layer or model-training technique.
