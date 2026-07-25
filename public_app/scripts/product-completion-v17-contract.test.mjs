import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const read = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("v17 client covers readiness, repair, support, and release contracts", async () => {
  const source = await read("src/lib/api.ts");
  for (const contract of [
    '"/setup/readiness"',
    '/setup/remediations/',
    '"/release/capability"',
    '"/release/status"',
    '"/dataset-repairs"',
    '/dataset-repairs/${encodeURIComponent(sessionId)}/issues',
    '/dataset-repairs/${encodeURIComponent(sessionId)}/plans',
    '/dataset-repair-plans/${encodeURIComponent(revisionId)}',
    '/dataset-repairs/${encodeURIComponent(sessionId)}/previews',
    '/dataset-repair-previews/${encodeURIComponent(previewId)}/publish',
    '/dataset-repair-revisions/${encodeURIComponent(revisionId)}',
    '/support-bundles/preview',
    '"/support-bundles"',
    'type WorkstationReadiness',
    'type DatasetRepairSession',
    'type DatasetRepairPreview',
    'type SupportBundle',
  ]) assert.match(source, new RegExp(escapeRegExp(contract)), contract);
});

test("setup gives one recommended action and never silently updates", async () => {
  const source = await read("src/routes/setup.tsx");
  assert.match(source, /Try a working example/);
  assert.match(source, /Train on your data/);
  assert.match(source, /Halo Forge will not install it automatically/);
  assert.match(source, /Technical details/);
});

test("guided repair is immutable, exact, paged, and separate from training", async () => {
  const source = await read("src/routes/datasets.repair.tsx");
  for (const behavior of [
    "original source unchanged",
    "Exact before and after preview",
    "Preview exact changes",
    "Publish repair",
    "did not publish a dataset or start training",
    "Continue to dataset",
    "Normalize chat roles",
    "Map label alias",
    "Correct media folder",
    "Previous",
    "Next",
  ]) assert.match(source, new RegExp(escapeRegExp(behavior)), behavior);
  assert.match(source, /source_index/);
  assert.match(source, /savedPlan/);
  assert.match(source, /savedRevision/);
  assert.match(source, /latest_preview_id/);
  assert.match(source, /ReadableValue/);
  assert.doesNotMatch(source, /<pre[^>]*>\{JSON\.stringify\(value/);
});

test("failed work and unresolved preflight expose support creation", async () => {
  const train = await read("src/routes/train.tsx");
  const activity = await read("src/components/shell/activity-center.tsx");
  assert.match(train, /Create support bundle/);
  assert.match(activity, /Create support bundle/);
  assert.match(train, /to="\/diagnostics"/);
  assert.match(activity, /href="\/diagnostics"/);
});

test("diagnostics previews removable support categories without upload", async () => {
  const diagnostics = await read("src/routes/diagnostics.tsx");
  assert.match(diagnostics, /selectedSupportCategories/);
  assert.match(diagnostics, /Create a support bundle you can inspect before sharing/);
  assert.match(diagnostics, /No upload was performed/);
  assert.match(diagnostics, /selectedSupportCategories\.length === 0/);
});

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
