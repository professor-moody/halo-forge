import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = async (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("v8 exposes reward-integrity and training-signal public types", async () => {
  const api = await source("src/lib/api.ts");
  for (const type of [
    "TrainingSignalCapabilityDescriptor",
    "TrainingRecordRef",
    "TrainingSignalSnapshot",
    "TrainingSignalShard",
    "RewardSystem",
    "RewardSystemRevision",
    "RewardSystemAuditor",
    "RewardAuditProtocolRevision",
    "RewardIntegrityProfileRevision",
    "RewardIntegrityBinding",
    "RewardIntegrityAudit",
    "RewardIntegrityObservation",
    "RewardIntegrityMetric",
    "RewardIntegrityDecision",
    "RewardIntegrityComparison",
    "RewardIntegrityComparisonPair",
    "ResolvedRewardBinding",
  ]) assert.match(api, new RegExp(`export type ${type}`), type);
});

test("v8 API covers capabilities, immutable resources, traces, audits, comparisons, and review", async () => {
  const api = await source("src/lib/api.ts");
  for (const path of [
    "/reward-integrity-capabilities",
    "/reward-systems",
    "/reward-audit-protocols",
    "/reward-integrity-profiles",
    "/training-signals",
    "/reward-integrity-audits",
    "/reward-integrity-audits/compare",
    "/review",
  ]) assert.match(api, new RegExp(path.replaceAll("/", "\\/")), path);
  assert.match(api, /runRewardIntegrityAudits/);
  assert.match(api, /runTrainingSignalShards/);
  for (const operation of [
    "validateRewardSystem",
    "createRewardAuditProtocol",
    "reviseRewardAuditProtocol",
    "createRewardIntegrityProfile",
    "reviseRewardIntegrityProfile",
  ]) assert.match(api, new RegExp(operation), operation);
  assert.match(api, /retryRewardIntegrityAudit: \(auditId: string, reason: string\)/);
  assert.match(api, /JSON\.stringify\(\{ reason \}\)/);
  assert.match(api, /compareRewardIntegrityAudits: \(baseAuditId: string, candidateAuditId: string, params:/);
  assert.match(api, /search\.set\("limit", String\(params\.limit\)\)/);
  assert.match(api, /search\.set\("offset", String\(params\.offset\)\)/);
});

test("Train and Experiments share one capability-driven audit binding", async () => {
  const binding = await source("src/components/research/reward-audit-binding.tsx");
  const train = await source("src/routes/train.tsx");
  const experiments = await source("src/routes/sweeps.tsx");
  for (const copy of ["Training verifier, sentinel, and reward mapping", "Same-sample capture protocol", "Integrity rules", "Fail pauses for review"]) assert.match(binding, new RegExp(copy));
  assert.match(binding, /capture_fidelity/);
  assert.match(binding, /aggregate_only/);
  assert.match(binding, /balanced_256/);
  assert.match(binding, /Optional checkpoint quality suite/);
  assert.match(binding, /listBenchmarkSuites/);
  assert.match(binding, /\["development", "unspecified"\]/);
  assert.match(binding, /protected evidence is never offered here/);
  assert.match(train, /RewardAuditBindingEditor/);
  assert.match(train, /reward_system_revision_id/);
  assert.match(experiments, /Boundaries & audits/);
  assert.match(experiments, /RewardAuditBindingEditor/);
  assert.match(experiments, /reward_audit_boundaries/);
  assert.match(train, /development_suite_revision_id/);
  assert.match(experiments, /development_suite_revision_id/);
  assert.doesNotMatch(train, /raw\.reward_development_suite_revision_id/);
  assert.doesNotMatch(experiments, /baseConfig\.reward_development_suite_revision_id/);
});

test("Run and Evaluate expose URL-restored training audit evidence", async () => {
  const run = await source("src/routes/runs.$runId.tsx");
  const evaluate = await source("src/routes/eval.tsx");
  const verifiers = await source("src/routes/verifiers.tsx");
  const workspace = await source("src/components/research/reward-integrity-workspace.tsx");
  assert.match(run, /RunIntegrityStrip/);
  assert.match(run, /Training audits/);
  for (const field of ["audit", "sample", "page", "classification"]) assert.match(run, new RegExp(`${field}\\?:`));
  assert.match(verifiers, /label: "Training audits"/);
  for (const view of ["profiles", "results", "compare"]) assert.match(workspace, new RegExp(`"${view}"`));
  for (const field of ["auditView", "auditBase", "auditCandidate", "auditSample", "auditPage", "auditClassification"]) assert.match(evaluate, new RegExp(field));
  assert.match(workspace, /Table equivalent/);
  assert.match(workspace, /Exact captured output/);
  assert.match(workspace, /Component traces/);
  assert.match(workspace, /Media evidence/);
  assert.match(workspace, /<audio controls/);
  assert.match(workspace, /Tool definitions/);
  assert.match(workspace, /change boundary/);
  assert.match(workspace, /BOUNDED EVIDENCE PAIRS/);
  assert.match(workspace, /Exact snapshot joins/);
  assert.match(workspace, /Stable-record joins · non-causal/);
  assert.match(workspace, /pairing_reason/);
  assert.match(workspace, /ComparisonPairInspector/);
  assert.match(workspace, /BASE AUDIT/);
  assert.match(workspace, /CANDIDATE AUDIT/);
  assert.match(workspace, /MobileEvidenceReview/);
  assert.match(workspace, /data-mobile-evidence-review/);
  assert.match(workspace, /Back to evidence list/);
  assert.match(workspace, /Previous evidence/);
  assert.match(workspace, /Next evidence/);
  assert.match(workspace, /h-11/);
  assert.match(workspace, /isTypingTarget/);
  assert.match(workspace, /role="button"/);
  assert.match(workspace, /tabIndex=\{0\}/);
  assert.match(workspace, /event\.key !== "Enter" && event\.key !== " "/);
  assert.match(workspace, /focus-visible:ring-2/);
  assert.match(workspace, /GUIDED REWARD SYSTEM/);
  assert.match(workspace, /definition: \{ optimizer_verifier_revision_id/);
});

test("operator actions remain reviewed and Activity links directly to reward evidence", async () => {
  const workspace = await source("src/components/research/reward-integrity-workspace.tsx");
  const activity = await source("src/components/shell/activity-center.tsx");
  const proposal = await source("src/components/review/review-proposal.tsx");
  for (const action of ["Continue", "Stop", "Fork"]) assert.match(workspace, new RegExp(`>${action}<`));
  assert.match(workspace, /Required decision reason/);
  assert.match(workspace, /Required retry reason/);
  assert.match(workspace, /retryRewardIntegrityAudit\(audit\.id, retryReason\.trim\(\)\)/);
  assert.match(workspace, /A proposal does not resolve this pause or start training/);
  assert.match(activity, /Open Training Audit/);
  assert.match(activity, /Compare Audit/);
  assert.match(activity, /reviewRewardIntegrityAudit/);
  assert.match(activity, /retryRewardIntegrityAudit\(domainId, reason\)/);
  assert.match(activity, /Paired coverage/);
  assert.match(proposal, /reward_integrity_audit/);
  assert.match(proposal, /Training audit evidence/);
  assert.match(proposal, /Protected purposes, test\/canary splits, and protected lineage are refused/);
});

test("reviewed Fork restores immutable checkpoint lineage in Train", async () => {
  const api = await source("src/lib/api.ts");
  const train = await source("src/routes/train.tsx");
  const workspace = await source("src/components/research/reward-integrity-workspace.tsx");
  const activity = await source("src/components/shell/activity-center.tsx");
  assert.match(api, /RewardIntegrityForkContext/);
  assert.match(api, /normalizeRewardIntegrityReviewResult/);
  assert.match(api, /rewardIntegrityForkContext/);
  assert.match(api, /\/fork-context/);
  assert.match(train, /fork_reward_audit/);
  assert.match(train, /applyRewardAuditForkContext/);
  assert.match(train, /source_reward_integrity_decision_id/);
  assert.match(train, /fork_checkpoint_hash/);
  assert.match(train, /Fork from reviewed checkpoint/);
  assert.match(workspace, /window\.location\.assign\(result\.href\)/);
  assert.match(activity, /window\.location\.assign\(result\.href\)/);
});
