import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const read = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("Dataset Lab API client covers source, version, and job lifecycles", async () => {
  const source = await read("src/lib/api.ts");
  for (const contract of [
    'request<{ items: DatasetRecordWire[] }>("/datasets")',
    'request<DatasetRecordWire>("/datasets"',
    '/datasets/${encodeURIComponent(datasetId)}/preview',
    '/datasets/${encodeURIComponent(datasetId)}/build',
    '/datasets/${encodeURIComponent(datasetId)}/versions',
    '/dataset-versions/${encodeURIComponent(versionId)}/preview',
    '/dataset-versions/${encodeURIComponent(versionId)}/export',
    '/dataset-versions/${encodeURIComponent(versionId)}/materialize',
    '/dataset-versions/${encodeURIComponent(versionId)}/clone-recipe',
    '/dataset-jobs/${encodeURIComponent(jobId)}/cancel',
    '/dataset-jobs/${encodeURIComponent(jobId)}/retry',
  ]) {
    assert.match(source, new RegExp(escapeRegExp(contract)), contract);
  }
});

test("Dataset Lab routes expose the full workbench and media-aware preview", async () => {
  const detail = await read("src/routes/datasets.$datasetId.tsx");
  const version = await read("src/routes/datasets.$datasetId.versions.$versionId.tsx");

  for (const label of ["Overview", "Preview", "Build", "Versions", "Build version", "Cancel", "Retry"]) {
    assert.match(detail, new RegExp(escapeRegExp(label)), label);
  }
  for (const recipeKind of ["dedup", "contamination", "failure_mining", "synthesize"]) {
    assert.match(detail, new RegExp(`\\b${recipeKind}\\b`), recipeKind);
  }
  assert.match(detail, /<img[\s\S]+<audio/, "image and audio previews");

  for (const action of ["Export", "Materialize assets", "Clone recipe", "Train"]) {
    assert.match(version, new RegExp(escapeRegExp(action)), action);
  }
  for (const evidence of ["Splits", "Statistics", "Rejections", "Contamination", "Provenance"]) {
    assert.match(version, new RegExp(escapeRegExp(evidence)), evidence);
  }
  assert.match(version, /datasetVersion:\s*versionId/);
  assert.match(version, /datasetSplit:\s*(activeSplit|trainingSplit)/);
});

test("Dataset Lab is part of desktop and mobile navigation", async () => {
  const sidebar = await read("src/components/shell/sidebar.tsx");
  const shell = await read("src/components/shell/index.tsx");
  const navigation = await read("src/components/shell/navigation.ts");
  assert.match(navigation, /to: "\/datasets", label: "Data"/);
  assert.match(sidebar, /PRIMARY_NAV\.map/);
  assert.match(shell, /PRIMARY_NAV\.map/);
});

test("Dataset Lab v2 API client covers artifact, comparison, evaluation, and mining lifecycles", async () => {
  const source = await read("src/lib/api.ts");
  for (const contract of [
    '/dataset-versions/${encodeURIComponent(versionId)}/training-artifacts',
    '/training-artifacts/${encodeURIComponent(artifactId)}',
    '/dataset-versions/${encodeURIComponent(versionId)}/runs',
    'other_version_id',
    '/runs/${encodeURIComponent(runId)}/launch-config',
    'request<{ items: BenchmarkSuite[] }>("/benchmark-suites")',
    '/benchmark-suites/${encodeURIComponent(suiteId)}/revisions',
    'request<Evaluation>("/evaluations"',
    '/evaluations/compare?',
    '/evaluation-mining/preview',
    '/evaluation-mining/build',
  ]) {
    assert.match(source, new RegExp(escapeRegExp(contract)), contract);
  }
});

test("Dataset Lab v2 dashboard closes the data, training, evaluation, and feedback loop", async () => {
  const version = await read("src/routes/datasets.$datasetId.versions.$versionId.tsx");
  const train = await read("src/routes/train.tsx");
  const run = await read("src/routes/runs.$runId.tsx");
  const evaluation = await read("src/routes/eval.tsx");

  for (const label of ["Trainer compatibility", "Exact token profile", "Training artifacts", "Compare versions", "Runs using this version"]) {
    assert.match(version, new RegExp(escapeRegExp(label)), label);
  }
  assert.match(version, /No registered trainer adapter accepts this version/);
  assert.match(version, /compatible_trainers/);

  for (const contract of ["datasetBindings", "parentRunId", "Managed dataset bindings", "Clone in Train", "output_root"]) {
    assert.match(train, new RegExp(escapeRegExp(contract)), contract);
  }
  for (const contract of ["preparing_dataset", "artifactPreparationJob", "api.datasetJob", "Preparing dataset artifact"]) {
    assert.match(train, new RegExp(escapeRegExp(contract)), contract);
  }
  for (const contract of ["Dataset bindings", "Evaluation history", "Clone in Train", "Evaluate"]) {
    assert.match(run, new RegExp(escapeRegExp(contract)), contract);
  }
  for (const contract of ["Benchmark suites", "Evaluation queue", "Base / candidate delta", "Build dataset from failures", "Legacy cohort"]) {
    assert.match(evaluation, new RegExp(escapeRegExp(contract)), contract);
  }
});

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
