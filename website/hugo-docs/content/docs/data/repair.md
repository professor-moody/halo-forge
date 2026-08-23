---
title: "Fix Training Data"
description: "Review deterministic, non-destructive fixes before publishing an immutable Dataset Version."
weight: 12
---

Choose **Fix data** when Own Data finds mapping errors, quarantined rows, broken
conversation structure, preference problems, duplicates, or missing media.
Halo Forge scans the complete source in the background and groups related
problems. Until that scan finishes, counts are labeled as estimates.

## What A Repair Does

A repair is an immutable overlay. Your original CSV, JSONL, Parquet file,
folder, image, and audio assets are never modified. Halo Forge preserves source
fingerprints, record identities, operator decisions, and before/after hashes.

Normal mode offers deterministic actions:

- update field mappings;
- parse safe scalar types;
- normalize chat roles and remove empty turns;
- map label aliases;
- correct a relative media root;
- provide an optional constant;
- edit a selected record;
- quarantine or explicitly exclude an unresolved record.

Halo Forge does not automatically rewrite free-form text with a model and does
not edit binary image or audio assets. A provider suggestion is generated only
when requested and counts as a repair only after you accept it.

## Review And Publish

The workspace shows original and repaired records side by side. **Preview
changes** runs an exact full-source pass and reports changed, accepted,
quarantined, duplicate, and split-impact counts. **Publish repair** creates an
immutable repair revision. Continue to Dataset Lab to build a new immutable
version; training never starts automatically.

If the source changes, Halo Forge marks the repair stale. **Rebase repair**
shows record-level conflicts and requires a new review.

## CLI

```bash
halo-forge data repair inspect ./training.csv --scenario-revision instruction-sft@1
halo-forge data repair preview REPAIR_SESSION_ID --trim
halo-forge data repair apply REPAIR_PREVIEW_ID
halo-forge data repair rebase REPAIR_SESSION_ID
```

The dashboard does not require JSON, internal IDs, source-file edits, or a
terminal. These commands provide automation parity for operators who want it.
