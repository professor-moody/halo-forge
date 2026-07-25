import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const read = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("v10 API client covers guidance, semantic readiness, corpus preparation, CPT, and v9 lifecycles", async () => {
  const source = await read("src/lib/api.ts");
  for (const contract of [
    '"/interface-capabilities"',
    '/training-scenarios',
    '"/training-scenarios/advise"',
    '"/training-scenario-examples"',
    '/training-scenarios/${encodeURIComponent(scenarioId)}/examples',
    'request<DatasetImportSession>("/dataset-imports"',
    '/dataset-imports/huggingface/options?',
    '/dataset-imports/${encodeURIComponent(importId)}/files',
    '/dataset-imports/${encodeURIComponent(importId)}/cancel',
    '/dataset-imports/${encodeURIComponent(importId)}/retry',
    '/files/${encodeURIComponent(fileId)}/content',
    '"Content-Range"',
    '/dataset-imports/${encodeURIComponent(importId)}/inspect',
    '/dataset-inspections/${encodeURIComponent(inspectionId)}/mapping-preview',
    '/dataset-inspections/${encodeURIComponent(inspectionId)}/semantic-preview?limit=',
    '/dataset-inspections/${encodeURIComponent(inspectionId)}/readiness',
    '/dataset-inspections/${encodeURIComponent(inspectionId)}/preparation-preview',
    '/dataset-inspections/${encodeURIComponent(inspectionId)}/register',
    '/dataset-sources/${encodeURIComponent(sourceId)}/refresh',
    '/dataset-versions/${encodeURIComponent(versionId)}/readiness',
    '/dataset-versions/${encodeURIComponent(versionId)}/proof-run',
    '/runs/${encodeURIComponent(runId)}/full-run',
    '"/document-extractors"',
    '"/document-extractions"',
    '/document-extractions/${encodeURIComponent(extractionId)}',
    '/document-extractions/${encodeURIComponent(extractionId)}/preview',
    '/dataset-versions/${encodeURIComponent(versionId)}/corpus-profile',
    '/dataset-versions/${encodeURIComponent(versionId)}/packing-plan',
    '"/cpt/preflight"',
    '"/cpt/launch"',
    'adaptation: "lora" | "full"',
    'budget_mode: "tokens" | "passes"',
    'max_sequence_length: number',
    'packing_plan?: CorpusPackingPlan | null',
    'work_item_id?: string | null',
    'type SemanticRecordPreview',
    'type ScenarioAdviceResult',
    'type GuidedExampleDescriptor',
    'type DatasetReadiness',
    'source_config?: string | null',
    'source_split?: string | null',
    'source_revision?: string | null',
    'resolved_revision?: string | null',
    'fingerprint?: string | null',
    'valid_records?: number | null',
    'invalid_records?: number | null',
    'disk_forecast?:',
    'capacity_override_reason?: string',
    'verifier_profile_revision_id',
    'verifierProfileRevisionId',
  ]) assert.match(source, new RegExp(escapeRegExp(contract)), contract);
});

test("guided studio exposes v10 advisor, gallery, semantic previews, readiness remediation, corpus preparation, and CPT controls", async () => {
  const source = await read("src/components/data/own-data-studio.tsx");
  for (const step of ["Goal", "Source", "Format", "Map", "Prepare", "Version", "Train"]) {
    assert.match(source, new RegExp(`label: "${step}"`), step);
  }
  for (const kind of ["direct", "constant", "concat", "nested_path", "conversation", "media_root"]) {
    assert.match(source, new RegExp(`kind: "${kind}"`), kind);
  }
  for (const behavior of [
    "Confirm this interpretation",
    "Confirm mapping",
    "Publish version",
    "Start proof run",
    "Start full run",
    "Semantic preview",
    "Advanced · technical record",
    "Advanced recipe",
    "Upload from this device",
    "Path on the workstation",
    "Browse configs and splits",
    "Choose a scenario manually",
    "Reviewed disk-capacity override reason",
    "Qualified training verifier",
    "No compatible qualified verifier is available",
    "Supported presentations",
    "Help me decide",
    "Find the best fit",
    "Try a working example",
    "Use this scenario",
    "Choose the document corpus",
    "Visible content",
    "Document boundaries",
    "Quarantine extraction failures",
    "Dataset readiness",
    "Recommended remediation",
    "Plan continued pretraining",
    "Adaptation method",
    "Choose LoRA or full",
    "Sequence length",
    "Token budget",
    "Corpus passes",
    "Packing",
    "Pack paragraphs with EOS boundaries",
    "Preparing the tokenizer-aware packing plan",
    "Track it in Activity",
    "Start continued pretraining",
    "corpus-adaptation@1",
    "paragraph_eos_non_overlap_v1",
  ]) assert.match(source, new RegExp(escapeRegExp(behavior)), behavior);
  assert.match(source, /useWorkspaceDraft/);
  assert.match(source, /pickDatasetSource/);
  assert.match(source, /4 \* 1024 \* 1024/);
  assert.match(source, /record\.uploaded_bytes \|\| 0/);
  assert.match(source, /Persist the durable import identity before transferring bytes/);
  assert.match(source, /sameStrings\(current\.selectedFileSignatures, signatures\)/);
  assert.match(source, /crypto\.subtle\.digest\("SHA-256"/);
  assert.match(source, /contentHash\);/);
  assert.match(source, /repairRevisionId/);
  assert.match(source, /kind: "repair_overlay"/);
  assert.match(source, /workspaceScrollContainer/);
  assert.match(source, /status === "preparing_dataset" \? 1_000 : false/);
  assert.match(source, /A preparation acknowledgement is a dataset job, not a training run/);
  assert.match(source, /Preparing the trainer-ready dataset/);
  assert.match(source, /qualification: "pass"/);
  assert.match(source, /verifier_profile_revision_id: selectedVerifierRevisionId/);
  assert.match(source, /api\.adviseTrainingScenario/);
  assert.match(source, /api\.guidedTrainingExamples/);
  assert.match(source, /api\.previewDatasetSemantics/);
  assert.match(source, /api\.datasetInspectionReadiness/);
  assert.match(source, /api\.documentExtractors/);
  assert.match(source, /api\.corpusProfile/);
  assert.match(source, /api\.corpusPackingPlan/);
  assert.match(source, /api\.cptPreflight/);
  assert.match(source, /api\.launchCpt/);
  for (const kind of ["chat", "preference", "tool", "vlm", "audio", "corpus"]) {
    assert.match(source, new RegExp(`item\\.kind === "${kind}"|kind === "${kind}"`), `semantic ${kind}`);
  }
  assert.match(source, /preparation: \{ \.\.\.fallback\.preparation, \.\.\.\(value\.preparation \?\? \{\}\) \}/);
  assert.match(source, /inspectionReadiness\?\.ready === false/);
  assert.match(source, /cptAdaptation: "" \| "lora" \| "full"/);
  assert.match(source, /cptBudgetMode: "tokens" \| "passes"/);
});

test("v18 recommends one immutable plan before model preparation and capacity work", async () => {
  const source = await read("src/components/data/own-data-studio.tsx");
  const api = await read("src/lib/api.ts");
  for (const behavior of [
    "Recommended plan",
    "Choose another compatible model",
    "Prepare and check",
    "Preparing the exact model",
    "Measuring this training plan",
    "Examples processed together",
    "Maximum text length",
    "Ready for a proof run",
    "Start proof run",
    "Technical details",
  ]) assert.match(source, new RegExp(escapeRegExp(behavior)), behavior);
  for (const method of [
    "recommendTrainingPlan",
    "chooseTrainingPlanAlternative",
    "confirmTrainingPlan",
    "prepareTrainingPlanModel",
    "createTrainingCapacityCheck",
    "retryModelPreparation",
    "retryTrainingCapacityCheck",
    "launchTrainingPlanProof",
  ]) assert.match(api, new RegExp(method), method);
  assert.match(source, /training_plan_revision_id: draft\.trainingPlanRevisionId/);
  assert.doesNotMatch(source, />Microbatch</);
  assert.doesNotMatch(source, />Gradient accumulation</);
});

test("normal guided mode uses semantic labels while technical JSON remains explicitly advanced", async () => {
  const source = await read("src/components/data/own-data-studio.tsx");
  assert.match(source, /scenarioKindLabel\(scenario\)/);
  assert.match(source, /Advanced · technical record/);
  assert.match(source, /Advanced recipe/);
  assert.doesNotMatch(source, /<Badge size="sm" tone="neutral">\{scenario\.canonical_shape\}<\/Badge>/);
  assert.doesNotMatch(source, />Source fingerprint<\/dt>/);
});

test("primary entry surfaces lead to guided own-data training and raw inputs stay advanced", async () => {
  const overview = await read("src/routes/index.tsx");
  const data = await read("src/routes/datasets.index.tsx");
  const guidedRoute = await read("src/routes/datasets.new.tsx");
  const train = await read("src/routes/train.tsx");
  const version = await read("src/routes/datasets.$datasetId.versions.$versionId.tsx");
  for (const source of [overview, data, train]) {
    assert.match(source, /Train on your data/);
    assert.match(source, /to="\/datasets\/new"/);
  }
  assert.match(overview, /Try a working example/);
  assert.match(train, /Try a working example/);
  assert.match(data, /Advanced register/);
  assert.match(train, /Advanced · roles, built-ins, and manual paths/);
  assert.match(train, /Speech recognition \(ASR\)/);
  assert.doesNotMatch(train, /<SelectItem value="classification">/);
  assert.doesNotMatch(train, /<SelectItem value="tts">/);
  assert.match(version, /Advanced · Training artifacts workbench/);
  assert.match(guidedRoute, /search\.example === 1/);
  assert.match(guidedRoute, /initialInspectionId=\{inspection\}/);
  assert.match(guidedRoute, /startWithExample=\{example === "1" && !inspection\}/);
});

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
