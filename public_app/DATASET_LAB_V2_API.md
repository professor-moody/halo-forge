# Dataset Lab v2 frontend API contract

The dashboard calls these resources below `/api/public`. Types live in
`src/lib/api.ts`; large renders, evaluations, and mined builds return a
persistent resource or job immediately.

| Method | Resource | Frontend use |
| --- | --- | --- |
| `GET`, `POST` | `/dataset-versions/{version_id}/training-artifacts` | List or render content-addressed trainer artifacts. |
| `GET` | `/training-artifacts/{artifact_id}` | Poll render state and resolved artifact evidence. |
| `GET` | `/dataset-versions/{version_id}/runs` | List runs bound to the immutable version. |
| `GET` | `/dataset-versions/{version_id}/compare?other_version_id=…` | Identity-aware version comparison. |
| `GET` | `/runs/{run_id}/launch-config` | Prefill Clone in Train with resolved config and bindings. |
| `GET`, `POST` | `/benchmark-suites` | List suites or create the first immutable revision. |
| `GET` | `/benchmark-suites/{suite_id}` | Read suite metadata and its revision history. |
| `GET` | `/benchmark-suite-revisions/{revision_id}` | Read one pinned immutable revision. |
| `POST` | `/benchmark-suites/{suite_id}/revisions` | Append an immutable suite revision. |
| `GET`, `POST` | `/evaluations` | List persistent evaluations or launch one. |
| `GET` | `/evaluations/{evaluation_id}` | Read progress, metrics, logs, and subject identity. |
| `GET` | `/evaluations/{evaluation_id}/samples` | Page standardized per-example evidence. |
| `GET` | `/evaluation-jobs` | List queued/running/interrupted evaluation jobs. |
| `POST` | `/evaluations/{evaluation_id}/cancel` | Cancel a queued or active evaluation. |
| `POST` | `/evaluations/{evaluation_id}/retry` | Retry from the last completed boundary. |
| `GET` | `/evaluations/compare?base_id=…&candidate_id=…` | Direction-aware metrics and record-level deltas. |
| `POST` | `/evaluation-mining/preview` | Preview reviewed failure selection and exclusions. |
| `POST` | `/evaluation-mining/build` | Queue an immutable failure-mined child version. |

Training preflight and launch accept `dataset_bindings[]` entries shaped as
`{ role, dataset_version_id, split }` plus optional `parent_run_id`.
`dataset_version_id` and `dataset_split` remain the train-only compatibility
shorthand. The dashboard also sends `output_root`; the API allocates the final
`<output_root>/<run_id>/` directory after minting the canonical run ID.

When managed data needs a trainer artifact, `/train/preflight` and
`/train/launch` return HTTP `202` with `status: "preparing_dataset"`, a reusable
`job_id`, and `artifact_preparation.job_url`. Launch also reserves the canonical
`run_id`. Poll the job, then retry the same request (preserving that `run_id` for
launch); the ready retry returns the normal HTTP `200` response and never
exposes test or canary data to the trainer.
