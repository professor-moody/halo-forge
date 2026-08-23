# Guided operational completion

Halo Forge now treats the end of one task as the beginning of one clear next
step. The normal interface assumes that you understand models and datasets, but
does not require knowledge of adapters, schemas, scheduler work items, evidence
classes, or statistical contrast definitions.

## After a proof run

Choose **Check training result**. Halo Forge verifies the proof artifact,
selects development evidence, evaluates the pinned base model and proof model
on the same immutable suite revision, and explains the result:

- **Ready to continue** — start the full run.
- **Review the tradeoff** — inspect examples before deciding.
- **Needs repair** — fix data or settings.
- **More evidence needed** — complete a development evaluation.
- **Training did not work** — open the fix guide.

Test, canary, operational, and holdout evidence is never used to guide this
decision. Continuing without compatible evidence requires a retained reason.

## Comparing approaches

The normal Studies workflow offers three choices:

- **Compare two approaches**
- **Try different data amounts**
- **Test data and method together**

Halo Forge uses repeat seeds 17, 42, and 101, keeps improvement and retention
evidence separate, and shows the number of runs plus available time and storage
estimates before launch. Factor definitions, planned contrasts, and exact
statistics remain available under Advanced.

## Creating examples from documents

Choose an immutable corpus version, select what to create, and choose:

- **Quick — 50**
- **Standard — 250**
- **Thorough — 1,000**

Halo Forge shows ten cited examples before launch. Generation runs in the
background and produces suggestions only. The next action is always **Review
examples**; generation never publishes a dataset or starts training.

## Specialized models

Specialized tasks are presented as goals:

- classify text, images, or audio;
- improve semantic search;
- rank search results.

Only combinations with a verified PyTorch optimizer, evaluation, artifact, and
replay contract appear as ready. Before **Use model** is shown, Halo Forge
reloads the artifact, checks its task and label or retrieval contract, and runs
a fixed-input inference.

## Local environments

Environment setup shows the template, success condition, model subject,
permissions, and episode count before launch. V16 environments are deterministic
and local: attempt-local files, local SQLite, and loopback fixture services may
be used; external writes are disabled.

**Replay the same actions** repeats the recorded trace exactly. **Run the model
again** invokes the selected local served model and produces separate evidence.
Trajectories enter Review Studio before publication or Dataset Lab handoff.

## Advanced details

Immutable IDs, hashes, exact evaluation revisions, work-item state, raw
manifests, planned contrasts, provider parameters, and lossless JSON/YAML remain
available in Technical details or Advanced. They are not required for the
guided workflows.
