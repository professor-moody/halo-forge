import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = async (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("adaptive research API covers policy, trajectory, gate, analysis, decision, and report lifecycles", async () => {
  const api = await source("src/lib/api.ts");
  for (const fragment of [
    "/checkpoint-policies",
    "/checkpoint-policies/resolve",
    "/trajectory",
    "/analyses",
    "/gate-decisions/",
    "/research-decisions",
    "/evidence-bundles",
    "/evaluations/history",
    "/evaluations/drift",
  ]) assert.match(api, new RegExp(fragment.replaceAll("/", "\\/")));
  assert.match(api, /checkpoint_policy_revision_id/);
  assert.match(api, /resolved_checkpoint_plan/);
});

test("experiment composer exposes guided checkpoints, evidence, and reviewed decisions", async () => {
  const experiments = await source("src/routes/sweeps.tsx");
  const workspace = await source("src/components/research/adaptive-workspace.tsx");
  for (const copy of ["Final only", "Periodic observation", "Guarded training", "Expected workstation load"]) assert.match(experiments, new RegExp(copy));
  for (const copy of ["Checkpoint trajectory", "Evidence contract", "Reviewed decision", "Export evidence"]) assert.match(workspace, new RegExp(copy));
  assert.match(workspace, /bootstrap_resamples:\s*10_000/);
  assert.match(workspace, /replicate_unit:\s*"seed"/);
  assert.match(workspace, /Pareto evidence/);
  assert.match(workspace, /Missing cells remain explicitly unavailable/);
  assert.match(workspace, /Override & continue/);
  assert.match(experiments, /Continue after complete evidence/);
  assert.match(experiments, /protect_evaluated:\s*true/);
  assert.match(experiments, /policy\.rules\.length === 0 && policy\.automatic_actions/);
});

test("workstation usability includes server drafts, searchable pickers, global lookup, and gate review", async () => {
  const api = await source("src/lib/api.ts");
  const evaluate = await source("src/routes/eval.tsx");
  const activity = await source("src/components/shell/activity-center.tsx");
  const palette = await source("src/components/shell/command-palette.tsx");
  const picker = await source("src/components/ui/search-picker.tsx");
  assert.match(api, /\/workspace-drafts\//);
  assert.match(api, /\/search\?/);
  assert.match(activity, /awaiting_review/);
  assert.match(activity, /Required review reason/);
  assert.match(palette, /api\.globalSearch/);
  assert.match(picker, /role="combobox"/);
  assert.match(picker, /ArrowDown/);
  assert.match(evaluate, /SearchPicker/);
  assert.match(evaluate, /Search datasets/);
  assert.match(evaluate, /Search ready versions/);
  assert.match(evaluate, /Advanced · use an unlisted identifier/);
});
