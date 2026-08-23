---
title: "Replay manifests"
description: "Reconstruct a captured launch and verify data, runtime, capacity, and real training-path identity with replay manifest v14."
weight: 50
---

`halo-forge replay <run_dir>` inspects a captured run and reconstructs its
launch command. Add `--launch` to run it again.

```bash
halo-forge replay ./models/raft
halo-forge replay ./models/raft --launch
```

Every shipped trainer writes `replay.json` beside `training_summary.json`.
Current manifests use format v14; v1-v13 remain readable.

## What v14 captures

Replay v14 records everything through v13, plus the exact real training-path
revision and certification. V13 added the managed runtime digest, host/device
identity, core qualification, and accelerator coexistence decision. V14 adds:

- pinned certification fixture and exact model commit;
- real trainer and capacity-adapter versions;
- before/after parameter evidence and reloadable artifact evidence;
- host identity and current path-certification identity; and
- workstation beta-qualification identity when that full evidence gate exists.

- canonical run ID, modality, model, timestamp, literal launch arguments,
  seeds, full resolved configuration, backend, platform, and package versions;
- local dataset hashes, pinned Hugging Face source identity, or immutable
  Dataset Lab role bindings and content-addressed trainer artifacts;
- immutable verifier profile, implementation, configuration, reward contract,
  qualification, and runtime identity;
- reward-system revision, ordered auditors, reward mapping, retention protocol,
  integrity profile, selected boundaries, signal capability, trace manifests,
  audit decisions, and runtime compatibility; and
- corpus extraction/version identity, tokenizer and packing hashes, budget
  semantics, LoRA/full adaptation mode, and CPT training-artifact identity;
- proof outcome and reviewed full-run decisions;
- controlled adaptation study protocols, arms, assignments, and deviations;
- specialized task, label-schema, model-head, processor, loss, and retrieval
  identities; and
- deterministic environment, fixture, tool, episode, snapshot, and trajectory
  identities.

Versioned sections are present only when the corresponding workflow was used.
Large datasets and signal traces are hashed or referenced, not embedded in the
manifest. Their checksum bundles remain in Dataset Lab and Reward Integrity
storage.

## Drift checks

Replay displays environment differences before showing the reconstructed
command. Launch behavior is stricter:

| Difference | Default | Explicit override |
|---|---|---|
| Python, platform, backend, or package environment | refuse | `--force` |
| Local dataset content | refuse | `--allow-dataset-drift` |
| Verifier identity/runtime | refuse | `--allow-verifier-drift --verifier-drift-reason "..."` |
| Reward system, auditors, protocol, profile, boundaries, capability, or trace identity | refuse | `--allow-reward-drift --reward-drift-reason "..."` |

Verifier and reward overrides require a reason. Halo Forge appends that reason
and the exact differences to `replay_overrides.jsonl`. The launch is then
intentional, but it is no longer an exact replay.

Legacy raw-verifier runs stay readable and carry `legacy_unqualified` rather
than fabricated qualification. Older runs without training-signal traces are
never described as reward-integrity monitored.

## Manifest versions

| Version | Added identity |
|---:|---|
| v1 | run, seed, config, dataset, environment, and arguments |
| v2 | managed dataset roles and trainer artifacts |
| v3 | immutable verifier reliability and qualification/runtime state |
| v4 | reward system, auditors, protocol, integrity profile, signal capability, traces, boundaries, and decisions |
| v5 | corpus extraction/version, tokenizer and packing, budget, adaptation, and CPT artifact identity |
| v6 | proof outcome and full-run decision |
| v7 | adaptation study protocol and assignments |
| v8 | specialized task and task-artifact contract |
| v9 | environment, episode-suite, snapshot, and trajectory identity |
| v10 | prepared evaluations, study assignments, grounding verification, environment execution, and reviewed decisions |
| v11 | immutable dataset repair, repaired-record set, source fingerprint, workstation readiness, and platform capability |
| v12 | immutable training plan, prepared-model commit, capacity evidence, safe adjustment, confirmation, and run binding |
| v13 | managed runtime, host/device, core qualification, occupancy, and contention evidence |
| v14 | real training-path certification, fixture, model commit, adapter versions, host, and workstation qualification |

The current replay format v14 uses `MANIFEST_VERSION = 14`. The loader supplies empty versioned
sections when reading older manifests,
preserving history without inferring evidence. A future-format manifest loads
with a warning; only fields the current client understands can participate in
its checks.

## Limits

Replay captures launch identity; it does not claim bit-for-bit numerical
equivalence across accelerator families. Reward Integrity separately audits the
exact retained outputs used during verifier-guided training. See
[Reward Integrity](/docs/reward-integrity/) for same-output sentinel behavior,
capture coverage, and reviewed pause rules.
