import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const read = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("the client exposes the real-path certification contract", async () => {
  const source = await read("src/lib/api.ts");
  for (const value of [
    "/runtime/paths?family=",
    "/training-path-revisions/${encodeURIComponent(pathRevisionId)}/certify",
    "/training-path-certifications/${encodeURIComponent(certificationId)}",
    'state: "runtime_ready" | "path_verified" | "verification_in_progress"',
    "recommended_path_revision_id",
    "runtime_ready",
    "beta_qualified",
  ]) assert.equal(source.includes(value), true, value);
});

test("Setup verifies core runtime and the recommended SFT path separately", async () => {
  const source = await read("src/routes/setup.tsx");
  for (const value of [
    'Prepare ${activeFamily === "rocm" ? "AMD" : "NVIDIA"} training',
    "Verify text training",
    "Verifying in Activity",
    "training-path-heading",
    "generic tensor update",
  ]) assert.equal(source.includes(value), true, value);
  assert.match(source, /recommended_path_revision_id/);
  assert.match(source, /api\.certifyTrainingPath/);
});

test("Train presents exactly the next safe path action", async () => {
  const source = await read("src/routes/train.tsx");
  for (const value of [
    "Hardware detection alone is not treated as training readiness",
    "Verify this training path",
    "Verifying in Activity",
    "Generic tensor checks do not unlock guided training",
  ]) assert.match(source, new RegExp(value));
  assert.match(source, /selectedTrainingPath\.state !== "path_verified"/);
});
