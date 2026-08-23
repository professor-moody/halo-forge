# Guided Dataset Repair

Halo Forge repairs training data through reviewed, immutable overlays. It never
changes the source file and never edits image or audio bytes.

The normal dashboard flow is:

```text
Inspect data → Fix data → review grouped issues → preview exact changes
             → publish repair → build immutable Dataset Version → proof run
```

`Fix data` appears from Own Data inspection, mapping failures, quarantined rows,
and proof-run remedies. The scan groups parse and delimiter problems, missing or
empty fields, incompatible scalar types, label aliases, invalid chat roles,
broken preference pairs, duplicates, and media-reference problems. A normal
screen shows one selected record with its original and proposed form side by
side.

Safe actions include field mapping, scalar parsing, chat-role normalization,
whitespace and empty-turn cleanup, label aliases, a corrected media root,
optional constants, selected-record edits, quarantine, and explicit exclusion.
Free-form model suggestions are opt-in and must be accepted individually.

Publication creates a content-addressed `DatasetRepairRevision`. Add it to an
ordered Dataset Lab recipe before mapping:

```yaml
steps:
  - kind: repair_overlay
    revision_id: REPAIR_REVISION_ID
  - kind: map
    fields:
      prompt: question
      response: answer
```

The build verifies the source fingerprint, before/after hashes, record
occurrence identity, and overlay checksum. If the source changes, the revision
becomes stale and `rebase` reports record-level conflicts rather than silently
reapplying edits.

CLI parity:

```bash
halo-forge data repair inspect ./training.csv --scenario-revision instruction-sft@1
halo-forge data repair preview REPAIR_SESSION_ID --trim --normalize-roles
halo-forge data repair apply REPAIR_PREVIEW_ID
halo-forge data repair rebase REPAIR_SESSION_ID
```

Use `--json` for automation. `--spec` is available for expert multi-action
plans; it is not required in the dashboard.

Repair, readiness, and platform capability identity are captured in Replay format v11
so a training run can be traced to the exact reviewed overlay.
