# Deterministic replay

`halo-forge replay <run_dir>` inspects a captured run and reconstructs its
launch command. Add `--launch` to run it again.

```bash
halo-forge replay ./models/raft
halo-forge replay ./models/raft --launch
```

Every shipped trainer writes `replay.json` beside `training_summary.json`.
Current manifests use format v14; v1-v13 remain readable.

## What v14 captures

Replay v14 retains the immutable v12 training-plan identity and v13 managed
runtime/occupancy identity. It adds the exact real-training-path revision and
certification evidence: fixture, model commit, trainer and capacity adapter
versions, host identity, and workstation beta-qualification identity. A core
runtime diagnostic is never represented as trainer certification.

```json
{
  "manifest_version": 14,
  "run_id": "run-01",
  "modality": "raft",
  "model_name": "Qwen/Qwen2.5-3B-Instruct",
  "seed": 42,
  "pythonhashseed": "42",
  "config": {},
  "dataset": {
    "kind": "managed_versions",
    "bindings": [
      {"role": "train", "dataset_version_id": "version-1", "split": "train"},
      {"role": "validation", "dataset_version_id": "version-1", "split": "validation"}
    ],
    "training_artifact": {
      "artifact_id": "training-artifact-1",
      "artifact_hash": "...",
      "adapter_id": "reasoning",
      "adapter_version": "3"
    }
  },
  "verifier": {
    "verifier_profile_revision_id": "verifier-revision-1",
    "implementation_fingerprint": "...",
    "sanitized_configuration_hash": "...",
    "legacy_unqualified": false
  },
  "reward_integrity": {
    "reward_system_revision_id": "reward-system-revision-1",
    "reward_system_hash": "...",
    "optimizer_verifier_revision_id": "verifier-revision-1",
    "auditors": [
      {"role": "primary_sentinel", "verifier_revision_id": "verifier-revision-2"}
    ],
    "protocol_revision_id": "balanced-256-revision-1",
    "integrity_profile_revision_id": "human-aligned-revision-1",
    "boundaries": [1, 4],
    "signal_capability": {"id": "raft:hf", "fidelity": "sampled"},
    "trace_manifests": [],
    "audit_decisions": []
  },
  "corpus_training": {
    "corpus_version": "version-corpus-1",
    "tokenizer_hash": "...",
    "packing_plan_hash": "...",
    "budget_mode": "passes",
    "corpus_passes": 1.0,
    "adaptation": "lora",
    "objective": "causal_next_token"
  },
  "training_outcome": {},
  "adaptation_study": {},
  "specialized_task": {},
  "agent_environment": {},
  "product_completion": {
    "dataset_repair_revision_id": "repair-revision-1",
    "repaired_record_set_hash": "...",
    "source_fingerprint": "...",
    "workstation_readiness_id": "readiness-1",
    "distribution_capability": {"platform": "darwin", "architecture": "arm64"}
  },
  "training_plan": {
    "revision_id": "plan-revision-1",
    "model_preparation_id": "model-preparation-1",
    "capacity_check_id": "capacity-check-1",
    "compute_shape_hash": "...",
    "resolved_model_commit": "...",
    "selected_adjustment": {"gradient_checkpointing": true},
    "confirmation": "confirmed"
  },
  "environment": {
    "python": "3.13.12",
    "platform": "Darwin arm64",
    "backend": "mlx",
    "packages": {}
  },
  "cli_args": ["raft", "train", "..."]
}
```

The manifest records:

- canonical run ID, modality, model, timestamp, and literal launch arguments;
- the training seed, `PYTHONHASHSEED`, full resolved configuration, backend,
  host platform, and relevant package versions;
- local dataset SHA-256 and size, pinned Hugging Face ID/revision, or immutable
  Dataset Lab role bindings and trainer-artifact identity;
- verifier profile revision, implementation and configuration fingerprints,
  reward contract, qualification scope, and runtime compatibility in v3+;
- reward-system, ordered auditor, reward-mapping, audit-protocol, integrity-
  profile, boundary, signal-capability, trace, audit-decision, and runtime
  identities in v4; and
- corpus extraction/version identity, tokenizer and packing hashes, budget
  semantics, adaptation mode, and CPT training-artifact identity in v5;
- proof outcome, base/candidate evaluation, finding summary, resource
  projection, and reviewed full-run decision identity in v6;
- adaptation study protocol, arm, factor, assignment, planned contrast, and
  deviation identity in v7;
- specialized task, label schema, model head, processor, loss adapter, and
  retrieval-corpus identity in v8; and
- deterministic environment, fixture, tool, episode-suite, snapshot,
  trajectory, verifier/reward, and model-subject identity through v10.

Versioned sections are present only when the corresponding workflow was used.
The manifest hashes or references large datasets and trace bundles instead of
embedding them. Those assets remain in Dataset Lab and Reward Integrity
storage, where their own checksum manifests are verified.

## Drift checks

Replay displays environment differences before showing the reconstructed
command. Launch behavior is deliberately stricter:

| Difference | Default | Explicit override |
|---|---|---|
| Python, platform, backend, or package environment | refuse launch | `--force` |
| Local dataset content hash | refuse launch | `--allow-dataset-drift` |
| Immutable verifier identity or runtime compatibility | refuse launch | `--allow-verifier-drift --verifier-drift-reason "..."` |
| Immutable reward system, auditors, protocol, profile, boundaries, capability, or trace identity | refuse launch | `--allow-reward-drift --reward-drift-reason "..."` |

Verifier and reward overrides require a non-empty reason. Halo Forge appends an
event containing the reason and exact differences to
`replay_overrides.jsonl` beside the manifest. An override makes the launch
intentional; it does not make the result an exact replay.

```bash
halo-forge replay ./models/raft --launch \
  --allow-reward-drift \
  --reward-drift-reason "Re-running against the newly qualified sentinel"
```

Legacy raw verifier launches remain replayable. They carry
`legacy_unqualified` rather than a fabricated qualification identity. Likewise,
v1-v3 runs without training-signal traces are readable but are never described
as reward-integrity monitored.

## Manifest versions

| Version | Added identity |
|---:|---|
| v1 | run, seed, config, dataset, environment, and launch arguments |
| v2 | managed Dataset Lab versions, role bindings, and trainer artifacts |
| v3 | immutable verifier reliability identity and qualification/runtime state |
| v4 | reward system, auditors, retention protocol, integrity profile, signal capability, traces, boundaries, and audit decisions |
| v5 | corpus extraction/version identity, tokenizer and packing identity, budget semantics, adaptation mode, and CPT artifact identity |
| v6 | proof outcome evidence, resource projection, and reviewed full-run decision |
| v7 | adaptation study protocol, arm, assignment, contrast, and deviation identity |
| v8 | specialized task, label schema, model head, processor, loss, and retrieval corpus |
| v9 | environment, fixtures, tools, episode suite, snapshots, and trajectories |
| v10 | prepared evaluations, study assignments, grounding verification, environment execution, and reviewed decisions |
| v11 | dataset repair revision and repaired-record set, source fingerprint, workstation-readiness snapshot, and resolved platform capability |
| v12 | immutable training plan, recommendation, model preparation, capacity evidence, safe adjustment, confirmation, and run binding |
| v13 | managed runtime digest, host/device identity, core qualification, and accelerator coexistence decision |
| v14 | real training-path revision/certification, fixture, exact model, adapter versions, host identity, and beta qualification |

Replay format v14 uses `MANIFEST_VERSION = 14`. The loader assigns empty versioned sections when an
older manifest lacks them, so historical runs
remain inspectable without implying evidence that was never recorded. A
manifest from a newer Halo Forge version loads with a warning; only fields
understood by the older client can be used.

## Dataset identity

Local dataset hashing streams the file so large JSONL sources do not need to fit
in memory. A one-byte change produces a different SHA-256. A Hugging Face source
records its ID and revision; pin a revision when exact source identity matters.

Managed datasets record every role (`train`, `validation`, and other supported
bindings), version and split, plus the content-addressed trainer artifact. Test
and canary rows are never silently converted into trainer inputs.

## Programmatic API

```python
from halo_forge.replay import (
    EnvironmentFingerprint,
    capture_manifest,
    compare_environments,
    compare_reward_identities,
    compare_verifier_identities,
    load_manifest,
    save_manifest,
)

manifest = capture_manifest(
    run_id="run-01",
    modality="raft",
    model_name="Qwen/Qwen2.5-3B-Instruct",
    seed=42,
    config=resolved_config,
    dataset_file=train_jsonl,
    verifier_binding=resolved_verifier,
    reward_integrity_binding=resolved_reward_system,
    cli_args=launch_args,
)
save_manifest(manifest, output_dir)

loaded = load_manifest(output_dir)
environment_diff = compare_environments(
    loaded.environment,
    EnvironmentFingerprint.capture().to_dict(),
)
```

## Storage and limits

```text
models/raft/
  replay.json
  replay_overrides.jsonl        # present only after an explicit drift override
  training_summary.json
  ...
```

Replay captures launch identity; it does not claim bit-for-bit numerical
equivalence across accelerator families. Runtime nondeterminism, kernel choice,
and unsupported precision metadata remain interpretation limits. Reward
Integrity separately verifies the exact retained outputs used for its audit;
see [REWARD_INTEGRITY.md](REWARD_INTEGRITY.md).
