import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);
const source = async (path) => readFile(new URL(path, root), "utf8");

test("V11 keeps proof outcomes and reviewed full-run decisions in Own Data", async () => {
  const ownData = await source("src/components/data/own-data-studio.tsx");
  const api = await source("src/lib/api.ts");
  for (const behavior of [
    "Check training result",
    "Review examples",
    "Start full run",
    "Continue anyway",
    "preparing the same development evidence",
  ]) {
    assert.ok(ownData.includes(behavior), behavior);
  }
  assert.match(api, /prepareTrainingOutcome/);
  assert.match(api, /actionableGuidance/);
  assert.match(api, /outcomeFindings/);
  assert.match(api, /fullRunContext/);
});

test("V12 exposes a bounded guided adaptation-study workspace", async () => {
  const experiments = await source("src/routes/sweeps.tsx");
  const api = await source("src/lib/api.ts");
  for (const behavior of [
    "Studies",
    "Compare two approaches",
    "Try different data amounts",
    "Test data and method together",
    "3 per approach",
    "Improvement + retention",
    "Time and storage estimate",
  ]) {
    assert.ok(experiments.includes(behavior), behavior);
  }
  assert.match(api, /createAdaptationStudy/);
  assert.match(api, /materializeAdaptationStudy/);
  assert.match(api, /analyzeAdaptationStudy/);
});

test("V13 keeps grounded generation separate from human review", async () => {
  const grounded = await source("src/routes/datasets.ground.tsx");
  const api = await source("src/lib/api.ts");
  for (const behavior of [
    "Create examples from documents",
    "Exact source spans are checked",
    "Examples are ready to review",
    "Review examples",
    "protected evidence is refused",
  ]) {
    assert.ok(grounded.includes(behavior), behavior);
  }
  assert.match(api, /createGroundingProfile/);
  assert.match(api, /createGroundingProfileRevision/);
  assert.match(api, /previewGroundedBatch/);
  assert.match(api, /createGroundingReviewProposal/);
});

test("V14 and V15 expose truthful task-model and local-environment paths", async () => {
  const train = await source("src/routes/train.tsx");
  const evaluate = await source("src/routes/eval.tsx");
  const models = await source("src/routes/models.tsx");
  const api = await source("src/lib/api.ts");
  assert.match(train, /Task models/);
  assert.match(evaluate, /Environments/);
  assert.match(evaluate, /\["suites", "launch", "results", "compare", "failure-review", "verifiers", "environments"\]/);
  assert.match(evaluate, /Replay the same actions/);
  assert.match(evaluate, /Run the model again/);
  assert.match(models, /Specialized task contract/);
  assert.match(models, /specialized_task\.model_head_hash/);
  assert.match(api, /specializedTasks/);
  assert.match(api, /createAgentEnvironment/);
  assert.match(api, /replayAgentEpisode/);
  assert.match(api, /publishAgentTrajectories/);
});
