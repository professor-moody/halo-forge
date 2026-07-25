import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const api = await readFile(new URL("../src/lib/api.ts", import.meta.url), "utf8");
const evalRoute = await readFile(new URL("../src/routes/eval.tsx", import.meta.url), "utf8");
const verifierRoute = await readFile(new URL("../src/routes/verifiers.tsx", import.meta.url), "utf8");
const trainRoute = await readFile(new URL("../src/routes/train.tsx", import.meta.url), "utf8");
const review = await readFile(new URL("../src/components/review/review-studio.tsx", import.meta.url), "utf8");
const reviewDesk = await readFile(new URL("../src/components/review/review-item-desk.tsx", import.meta.url), "utf8");
const reviewProposal = await readFile(new URL("../src/components/review/review-proposal.tsx", import.meta.url), "utf8");
const activityCenter = await readFile(new URL("../src/components/shell/activity-center.tsx", import.meta.url), "utf8");

test("v7 exposes the verifier reliability public types", () => {
  for (const type of [
    "VerifierCapabilityDescriptor",
    "VerifierProfile",
    "VerifierProfileRevision",
    "VerifierRevisionComponent",
    "VerifierRewardContract",
    "ResolvedVerifierBinding",
    "VerifierCalibrationProtocolRevision",
    "VerifierCalibration",
    "VerifierCalibrationSample",
    "VerifierCalibrationMetric",
    "VerifierQualificationProfileRevision",
    "VerifierQualificationDecision",
    "VerifierAlias",
    "VerifierCalibrationComparison",
  ]) assert.match(api, new RegExp(`export type ${type}`), type);
});

test("v7 API client covers profile, calibration, qualification, usage, and bounded evaluation operations", () => {
  for (const path of [
    "/verifier-reliability/capabilities",
    "/verifier-profiles",
    "/verifier-calibration-protocols",
    "/verifier-qualification-profiles",
    "/verifier-calibrations",
    "/verifier-qualifications",
    "/qualify",
    "/evaluation-batches",
    "/comparison-samples",
    "/runtime-compatibility",
    "/usage",
  ]) assert.match(api, new RegExp(path.replace(/[/-]/g, (value) => `\\${value}`)), path);
});

test("Evaluate has six task-oriented destinations and a five-view Verifier Studio", () => {
  for (const label of ["Suites", "Launch", "Results", "Compare", "Failure Review", "Verifiers"]) assert.match(evalRoute, new RegExp(`label: "${label}"`));
  for (const label of ["Catalog", "Profiles", "Calibrate", "Compare", "Qualification"]) assert.match(verifierRoute, new RegExp(`label: "${label}"`));
  assert.match(verifierRoute, /GUIDED PROFILE/);
  assert.match(verifierRoute, /GUIDED CALIBRATION/);
  assert.match(verifierRoute, /component_trace/);
  assert.match(verifierRoute, /Qualification reasons/);
  assert.match(verifierRoute, /Append qualification decision/);
});

test("comparison launch and review navigation are bounded and URL-restorable", () => {
  assert.match(evalRoute, /slice\(0, 4\)/);
  assert.match(evalRoute, /launchEvaluationBatch/);
  assert.match(evalRoute, /candidates: candidates\.map/);
  assert.match(evalRoute, /verifier_profile_revision_id: effectiveVerifierRevisionId/);
  assert.match(evalRoute, /Qualified verifier · required/);
  assert.match(evalRoute, /same immutable verifier revision is bound to every subject/i);
  assert.match(api, /verifier_profile_revision_id\?: string/);
  assert.match(evalRoute, /record\?: string/);
  assert.match(review, /reviewQueueSummaries/);
  assert.match(review, /reviewItemNeighbors/);
  assert.match(review, /limit: 100/);
  assert.doesNotMatch(review, /limit: 500/);
  assert.doesNotMatch(review, /limit: 300/);
  assert.doesNotMatch(review, /slice\(0, 50\)/);
});

test("reviewed dataset preview is semantic instead of raw JSON", () => {
  assert.match(reviewDesk, /SemanticDatasetPreview/);
  for (const label of ["Added", "Removed", "Replaced", "Quarantined", "Split affected", "Media overlap"]) assert.match(reviewDesk, new RegExp(`label="${label}"`));
  assert.doesNotMatch(reviewDesk, /JSON\.stringify\(preview\.data/);
});

test("development calibration failures open as guided reviewed acquisition proposals", () => {
  assert.match(reviewProposal, /verifier_calibration/);
  for (const selector of [
    "false_accept",
    "false_reject",
    "high_confidence_disagreement",
    "repeat_instability",
    "order_flip",
    "ranking_inversion",
    "threshold_adjacent",
    "parser_runtime",
    "subgroup",
    "chain_component",
  ]) assert.match(reviewProposal, new RegExp(selector), selector);
  for (const field of ["range_fraction", "tolerance", "subgroupKey", "componentRevisionId"]) assert.match(reviewProposal, new RegExp(field), field);
  assert.match(reviewProposal, /Development-only evidence/);
  assert.match(reviewProposal, /selector: draft\.verifierSelector/);
  assert.match(reviewProposal, /options: verifierSelectorOptions\(draft\)/);
  assert.match(verifierRoute, /Open Review Proposal/);
  assert.match(activityCenter, /Open Review Proposal/);
});

test("calibration diagnostics read semantic metadata from the primary metric", () => {
  for (const detail of ["confusion_matrix", "per_class", "per_label", "threshold_curve", "false_accept_rate", "false_reject_rate"]) assert.match(verifierRoute, new RegExp(detail), detail);
  assert.match(verifierRoute, /Report only · no threshold is applied automatically/);
});

test("training binds exact qualified verifier revisions while preserving advanced legacy inputs", () => {
  assert.match(trainRoute, /Qualified verifier profile/);
  assert.match(trainRoute, /verifier_profile_revision_id/);
  assert.match(trainRoute, /legacy raw verifier/);
  assert.match(trainRoute, /legacy unqualified/);
  assert.match(trainRoute, /c\.verifierProfileRevisionId \? undefined : c\.verifier/);
});

test("guided verifier profiles require pinned model-backed identity and capability-compatible contracts", () => {
  assert.match(verifierRoute, /Pinned model revision/);
  assert.match(verifierRoute, /isPinnedModelRevision\(judgeModelRevision\)/);
  assert.match(verifierRoute, /Moving aliases cannot be published in Guided mode/);
  assert.match(verifierRoute, /\["verified", "valid"\]\.includes\(item\.integrity/);
  assert.match(verifierRoute, /rewardArtifactOptions\.some/);
  assert.match(verifierRoute, /selected\.modalities\.includes\(modality\)/);
  assert.match(verifierRoute, /selected\.task_types\.includes\(taskType\)/);
  assert.match(verifierRoute, /Only loadable final, merged, converted, or quantized occurrences with verified content identity are shown/);
  assert.match(verifierRoute, /captured from the current Python, toolchain, and relevant hardware/);
});

test("the client adapts canonical v7 service identities without weakening exact bindings", () => {
  assert.match(api, /qualified_only/);
  assert.match(api, /verifier_profile_revision_id/);
  assert.match(api, /source_kind === "label_set_revision" \? "label_set"/);
  assert.match(api, /search\.set\("partition", params\.split\)/);
  assert.match(api, /normalizeVerifierCalibration/);
  assert.match(api, /normalizeVerifierProfile/);
  assert.match(api, /kind: family === "reward_model" \? "artifact"/);
  assert.match(api, /ref: jsonString\(reference/);
  assert.match(verifierRoute, /tie_policy: "error"/);
  assert.match(verifierRoute, /error_behavior: "fail_closed"/);
  assert.doesNotMatch(verifierRoute, /tie_policy: "allow"/);
  assert.doesNotMatch(verifierRoute, /error_behavior: "propagate"/);
});
