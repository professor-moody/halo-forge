# Halo Forge 2.0.0-alpha-2 Release Notes

Halo Forge 2.0.0-alpha-2 is the repository's next macOS desktop developer
preview. The release workflow may attach a clearly named, checksummed unsigned
DMG to this prerelease. It is not notarized or a trusted normal installer; CLI
and local-browser installation remain the supported public paths. If signing
credentials become available, the same workflow can instead produce and
validate a supported signed/notarized candidate.

## Highlights

- Target: an unsigned, checksummed macOS Apple Silicon developer preview with
  truthful release metadata; signed/notarized promotion remains a funded gate.
- **Train on your data** now provides guided scenario advice, semantic record
  previews, action-oriented readiness checks, and reviewed example sources
  across the desktop and browser surfaces.
- The new document-corpus path extracts visible text with provenance and
  quarantined failures, publishes an immutable corpus version, renders exact
  tokenizer-aware packing, and launches explicit LoRA or full continued
  pretraining through PyTorch or verified native MLX.
- Proof runs now lead into a reviewed outcome assessment before a full run,
  with separate technical and development-quality evidence.
- Experiments includes bounded adaptation studies; Data includes grounded
  corpus generation with exact citations and Review Studio handoff.
- Guided Own Data now includes verified classification, multi-label,
  embedding, reranking, image-classification, and audio-classification paths.
- Evaluate includes deterministic local environments, exact trace replay, and
  reviewed trajectory publication for existing trainers.
- The release workflow validates packaged-runtime smoke for every DMG. When
  credentials exist, it additionally requires codesign, notarization stapling,
  and Gatekeeper acceptance before marking the artifact supported.
- An unsigned prerelease asset includes `-unsigned-preview` in its filename, a
  SHA-256 checksum, and a manifest recording that it is unsupported.
- The website recommends a DMG only after the signed/notarized gates pass.

## Release-candidate hardening

- Development, CI, and desktop packaging now share a reviewed `uv.lock` and
  generated standard/MLX pip constraints. Dependency drift fails the release
  gate, while wheel consumers retain bounded compatibility ranges.
- Ruff undefined-name checks are blocking. This includes the Review Studio
  validation path that previously swallowed a missing `validate_record`
  import and could reject valid reviewed records.
- Chat-template identity now has one versioned digest contract with distinct
  absent, empty, unreadable, and unsupported states plus conversion, catalog,
  and legacy-record round-trip coverage.
- The reasoning modality smoke now persists canonical cycle checkpoints, so
  its checkpoint and resume evidence matches the real trainer contract without
  rewriting the accepted baseline.
- The full Python suite runs against disposable per-test Halo Forge state;
  cached databases, workers, and auth stores cannot leak into the operator's
  real `~/.halo-forge` directory or across cases.
- Sandbox and loopback tests probe whether the capability can actually execute,
  not merely whether a host binary exists. Unsupported parent sandboxes are
  reported as explicit capability skips while policy failure paths still run.
- The packaged macOS runtime has a frozen dependency build, embedded runtime
  self-check, real two-step tiny-model SFT proof, dashboard health check, and
  route smoke. The unsigned preview path records those completed gates without
  claiming Developer ID trust. Signing, notarization, and stapling remain
  mandatory before promotion to a supported normal installer.

## Install

For the supported path, install from source and use the CLI or local browser
dashboard. Informed testers may use the unsigned DMG from the official GitHub
prerelease after verifying its checksum and following the
[developer-preview instructions](https://halo-forge.io/docs/getting-started/install-desktop/#unsigned-macos-developer-preview).

## Notes

- Linux desktop packages remain workflow artifacts until the public Linux install path is finalized.
- CLI/source installation remains documented in the Quick Start.
- The old `2.0.0-alpha-1` DMG was unsigned and may be rejected by macOS Gatekeeper.
