# Halo Forge 2.0.0-alpha-2 Release Notes

Halo Forge 2.0.0-alpha-2 is the repository's next macOS desktop release
candidate. It is not a public desktop release until the release workflow has
produced, signed, notarized, verified, and uploaded the named artifacts. The
currently published GitHub prerelease is the unsigned alpha-1 developer-test
build; CLI and local-browser installation remain the normal public path.

## Highlights

- Target: a signed and notarized macOS Apple Silicon DMG for browser downloads.
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
- The release workflow must validate codesign, notarization stapling,
  Gatekeeper acceptance, and packaged-runtime smoke before upload.
- The future GitHub release must include the DMG, SHA-256 checksum, and release
  manifest.
- The website must only link the artifact after those gates pass.

## Install

Until alpha-2 is actually published, install from source and use the CLI or
local browser dashboard. After publication, the canonical download page will
be `https://halo-forge.io/download/`.

## Notes

- Linux desktop packages remain workflow artifacts until the public Linux install path is finalized.
- CLI/source installation remains documented in the Quick Start.
- The old `2.0.0-alpha-1` DMG was unsigned and may be rejected by macOS Gatekeeper.
