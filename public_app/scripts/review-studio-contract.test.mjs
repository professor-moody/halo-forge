import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("Review Studio client covers schema, acquisition, review, suggestion, and label-set lifecycles", async () => {
  const api = await source("src/lib/api.ts");
  for (const contract of [
    "/review-capabilities",
    "/annotation-schemas",
    "/annotation-schemas/validate",
    "/annotation-schema-revisions/",
    "/acquisition-batches",
    "/review-queues",
    "/statistics",
    "/event-batches",
    "/review-items/",
    "/suggestions",
    "/label-set-revisions/",
    "/dataset-preview",
    "/dataset-build",
    "/spec-descriptors/",
  ]) assert.match(api, new RegExp(escapeRegExp(contract)), contract);
  for (const type of ["AnnotationSchemaRevision", "AcquisitionSource", "AcquisitionRequest", "AcquisitionBatch", "ReviewQueue", "ReviewItem", "ReviewEvent", "LabelSetRevision", "DatasetBuildPreview", "SpecDescriptor"]) assert.match(api, new RegExp(`export type ${type}`), type);
});

test("Data navigation and contextual evidence links open a guided review proposal", async () => {
  const tabs = await source("src/components/data/data-section-tabs.tsx");
  const evaluation = await source("src/routes/eval.tsx");
  const version = await source("src/routes/datasets.$datasetId.versions.$versionId.tsx");
  const run = await source("src/routes/runs.$runId.tsx");
  const models = await source("src/routes/models.tsx");
  const route = await source("src/routes/datasets.review.tsx");
  assert.match(tabs, /Datasets/);
  assert.match(tabs, /Review queues/);
  assert.match(route, /baseRef/);
  assert.match(evaluation, /Review these examples/);
  assert.match(evaluation, /source: "evaluation_comparison"/);
  assert.match(version, /source: "dataset_version"/);
  assert.match(run, /source: "run_samples"/);
  assert.match(models, /source: "playground_session"/);
});

test("Proposal workflow is structured, recoverable, deterministic, and eligibility-aware", async () => {
  const proposal = await source("src/components/review/review-proposal.tsx");
  const editor = await source("src/components/ui/structured-spec-editor.tsx");
  for (const copy of ["Select the evidence", "Define the decision", "Set the review policy", "Confirm the review set", "Prepare candidates"]) assert.match(proposal, new RegExp(copy));
  for (const strategy of ["candidate_failure", "regression", "verifier_disagreement", "low_score", "low_margin", "coverage_gap", "diversity", "random", "explicit"]) assert.match(proposal, new RegExp(strategy));
  assert.match(proposal, /Held-out.*(isolated|refused)/i);
  assert.match(proposal, /useWorkspaceDraft/);
  assert.match(proposal, /seed: 42/);
  assert.match(proposal, /api\.acquisitionBatch/);
  assert.match(proposal, /refetchInterval/);
  assert.match(proposal, /batch\.status === "ready"/);
  assert.match(proposal, /Activity Center/);
  assert.match(proposal, /Retry preparation/);
  assert.match(proposal, /StructuredSpecEditor/);
  assert.match(editor, /Advanced/);
  assert.match(editor, /validateSpecDescriptor/);
  assert.match(editor, /visible_when/);
});

test("Focused reviewer supports multimodal evidence, two-pass state, drafts, keyboard flow, and conflict safety", async () => {
  const desk = await source("src/components/review/review-item-desk.tsx");
  const studio = await source("src/components/review/review-studio.tsx");
  for (const evidence of ["Tool definitions", "Option A", "MediaImage", "MediaAudio", "<audio", "messages"]) assert.match(desk, new RegExp(escapeRegExp(evidence)), evidence);
  for (const state of ["Blind second pass", "Adjudication", "conflict", "start-second-pass"]) assert.match(`${desk}\n${studio}`, new RegExp(escapeRegExp(state), "i"), state);
  assert.match(desk, /useWorkspaceDraft/);
  assert.match(desk, /expected_active_event_id/);
  assert.match(desk, /error\.status === 409/);
  assert.match(desk, /idempotency_key/);
  assert.match(desk, /eventType: "defer"/);
  assert.match(desk, /pass_2_flip_candidates/);
  assert.match(desk, /window\.addEventListener\("keydown"/);
  assert.match(desk, /fixed inset-x-0 bottom-0/);
  assert.match(desk, /env\(safe-area-inset-bottom\)/);
  assert.match(desk, /Reveal suggestion/);
  assert.match(desk, /Enlarged review image/);
  assert.match(desk, /Candidate ranking/);
  assert.match(desk, /Move candidate/);
  for (const field of ["Messages", "Tools", "Expected calls", "Expected results", "Add argument"]) assert.match(desk, new RegExp(field));
  assert.doesNotMatch(desk, /Ordered ranking JSON|Structured correction JSON/);
});

test("Reviewed labels verify and preview before explicit Dataset Lab publication", async () => {
  const desk = await source("src/components/review/review-item-desk.tsx");
  for (const action of ["Publish label set", "Verify integrity", "Dataset handoff", "Preview", "Build"]) assert.match(desk, new RegExp(action), action);
  assert.match(desk, /api\.reviewCapabilities/);
  assert.match(desk, /build_modes/);
  assert.match(desk, /materialize_assets/);
  assert.match(desk, /parent_version_id/);
  assert.match(desk, /Publishing label set/);
  assert.match(desk, /refetchInterval/);
  assert.match(desk, /publication\.work_item_id/);
});

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
