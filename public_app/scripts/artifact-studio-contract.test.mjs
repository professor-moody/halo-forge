import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const modelsSource = await readFile(new URL("../src/routes/models.tsx", import.meta.url), "utf8");
const apiSource = await readFile(new URL("../src/lib/api.ts", import.meta.url), "utf8");
const runSource = await readFile(new URL("../src/routes/runs.$runId.tsx", import.meta.url), "utf8");
const versionSource = await readFile(new URL("../src/routes/datasets.$datasetId.versions.$versionId.tsx", import.meta.url), "utf8");

test("conversion selector exposes only formats verified by the current backend", () => {
  const values = [...modelsSource.matchAll(/<option value="(huggingface|mlx|gguf|onnx)">/g)].map((match) => match[1]);
  assert.deepEqual([...new Set(values)], ["huggingface", "mlx", "gguf"]);
  assert.doesNotMatch(modelsSource, /<option value="onnx">/);
  assert.match(modelsSource, /post-training quantization, not quantization-aware training/i);
});

test("operator approvals carry review context", () => {
  assert.match(apiSource, /review_note: reviewNote/);
  assert.match(modelsSource, /Override note · required/);
  assert.match(modelsSource, /Why this cleanup is safe to approve/);
});

test("durable sessions and structured qualification profiles use public API resources", () => {
  assert.match(apiSource, /listPlaygroundSessions:/);
  assert.match(apiSource, /appendPlaygroundMessage:/);
  assert.match(apiSource, /reviewPlaygroundSession:/);
  assert.match(apiSource, /message_ids\?: string\[\]/);
  assert.match(apiSource, /createQualificationProfile:/);
  assert.match(modelsSource, /Benchmark draft/);
  assert.match(modelsSource, /Data source draft/);
});

test("Playground constructs explicit persisted base and candidate preference pairings", () => {
  for (const label of ["Persisted user prompt", "Base assistant response", "Candidate assistant response", "Create preference review queue"]) assert.match(modelsSource, new RegExp(label));
  assert.match(modelsSource, /kind: "review_queue"/);
  assert.match(modelsSource, /pairings: \[pairing\]/);
  assert.match(modelsSource, /base\.message\.content === candidate\.message\.content/);
  assert.match(modelsSource, /revision\.task_type === "pairwise" \|\| revision\.task_type === "ranking"/);
  assert.doesNotMatch(modelsSource, /Preference pairing JSON/);
});

test("task-oriented detail views keep URL-addressable tabs", () => {
  for (const label of ["Monitor", "Metrics", "Data", "Evaluation", "Artifacts", "Logs"]) assert.match(runSource, new RegExp(`label: "${label}"`));
  for (const label of ["Overview", "Records", "Quality", "Training", "Lineage"]) assert.match(versionSource, new RegExp(`label: "${label}"`));
  assert.match(versionSource, /view: tab\.id/);
});
