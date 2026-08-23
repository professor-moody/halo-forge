import { Link } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowLeft,
  ArrowRight,
  AudioLines,
  BookOpen,
  Braces,
  Check,
  CheckCircle2,
  ChevronDown,
  CircleDashed,
  Cloud,
  Code2,
  Database,
  FileText,
  FileJson,
  FolderOpen,
  HelpCircle,
  Image,
  Layers3,
  ListChecks,
  Loader2,
  MessageSquareText,
  Play,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  UploadCloud,
  WandSparkles,
  Wrench,
  XCircle,
  type LucideIcon,
} from "lucide-react";
import {
  useEffect,
  useMemo,
  useState,
  type DragEvent,
  type ReactNode,
} from "react";
import { parse as parseYaml } from "yaml";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  ApiError,
  api,
  connectionMode,
  type CorpusPackingPlan,
  type CorpusPackingPlanResponse,
  type CorpusPackingRequest,
  type CorpusProfile,
  type CorpusTrainingConfig,
  type DatasetImportSession,
  type DatasetPreparationPlan,
  type DatasetReadiness,
  type DatasetRecipe,
  type DatasetSourceInspection,
  type FieldMappingExpression,
  type FieldMappingPlan,
  type GuidedExampleDescriptor,
  type MappingPreview,
  type ScenarioAdviceRequest,
  type ScenarioAdviceResult,
  type SchemaCandidate,
  type SemanticPreviewResponse,
  type SemanticRecordPreview,
  type TrainingMode,
  type TrainingScenarioDescriptor,
  type TrainingScenarioExample,
  type TrainingScenarioField,
} from "@/lib/api";
import { isDesktopRuntime, pickDatasetSource } from "@/lib/desktop-bridge";
import { useBackendInfo, useDatasetInspection, useInterfaceCapabilities, useTelemetry, useTrainingModels, useTrainingScenarios, useWorkspaceInfo } from "@/lib/hooks";
import { useWorkspaceDraft } from "@/lib/workspace-draft";
import { cn } from "@/lib/utils";

const STEPS = [
  { id: "goal", label: "Goal", detail: "What should the model learn?" },
  { id: "source", label: "Source", detail: "Where is the data?" },
  { id: "format", label: "Format", detail: "Confirm what we detected." },
  { id: "map", label: "Map", detail: "Connect fields to training." },
  { id: "prepare", label: "Prepare", detail: "Clean, split, and check." },
  { id: "version", label: "Version", detail: "Publish immutable data." },
  { id: "train", label: "Train", detail: "Prove the path first." },
] as const;

type StudioStep = (typeof STEPS)[number]["id"];
type SourceMode = "desktop" | "upload" | "workstation" | "huggingface" | "example";

type PreparationSettings = {
  normalizeWhitespace: boolean;
  validate: boolean;
  quarantineInvalid: boolean;
  exactDedup: boolean;
  fuzzyDedup: boolean;
  trainRatio: number;
  validationRatio: number;
  testRatio: number;
  contamination: boolean;
  groupMedia: boolean;
  preserveDocumentBoundaries: boolean;
  preserveHeadings: boolean;
  stripBoilerplate: boolean;
  quarantineExtractionFailures: boolean;
};

type OwnDataDraft = {
  step: StudioStep;
  scenarioId: string;
  scenarioRevisionId: string;
  candidateScenarioRevisionId: string;
  candidateConfirmed: boolean;
  sourceMode: SourceMode;
  sourcePath: string;
  repairRevisionId: string;
  huggingFaceId: string;
  huggingFaceConfig: string;
  huggingFaceSplit: string;
  huggingFaceRevision: string;
  exampleId: string;
  selectedFileNames: string[];
  selectedFileSignatures: string[];
  capacityOverrideReason: string;
  importId: string;
  inspectionId: string;
  mappingPlan: FieldMappingPlan | null;
  mappingConfirmed: boolean;
  preparation: PreparationSettings;
  advancedRecipe: boolean;
  rawRecipe: string;
  datasetName: string;
  datasetDescription: string;
  datasetId: string;
  buildJobId: string;
  versionId: string;
  trainerMode: string;
  model: string;
  verifierRevisionId: string;
  trainingPlanId: string;
  trainingPlanRevisionId: string;
  modelPreparationId: string;
  capacityCheckId: string;
  proofRunId: string;
  outcomeAssessmentId: string;
  outcomeOverrideReason: string;
  fullRunId: string;
  advisorGoal: string;
  advisorModality: string;
  advisorSourceLayout: string;
  cptAdaptation: "" | "lora" | "full";
  cptMaxSequenceLength: number;
  cptBudgetMode: "tokens" | "passes";
  cptTargetTokens: number;
  cptCorpusPasses: number;
  cptPacking: string;
};

const DEFAULT_PREPARATION: PreparationSettings = {
  normalizeWhitespace: true,
  validate: true,
  quarantineInvalid: true,
  exactDedup: true,
  fuzzyDedup: false,
  trainRatio: 80,
  validationRatio: 10,
  testRatio: 10,
  contamination: true,
  groupMedia: true,
  preserveDocumentBoundaries: true,
  preserveHeadings: true,
  stripBoilerplate: true,
  quarantineExtractionFailures: true,
};

const CORPUS_PREPARATION: PreparationSettings = {
  ...DEFAULT_PREPARATION,
  fuzzyDedup: true,
  trainRatio: 90,
  validationRatio: 10,
  testRatio: 0,
};

function defaultDraft(example = false, inspectionId = ""): OwnDataDraft {
  return {
    step: inspectionId ? "format" : "goal",
    scenarioId: "",
    scenarioRevisionId: "",
    candidateScenarioRevisionId: "",
    candidateConfirmed: false,
    sourceMode: example ? "example" : isDesktopRuntime() ? "desktop" : "upload",
    sourcePath: "",
    repairRevisionId: "",
    huggingFaceId: "",
    huggingFaceConfig: "",
    huggingFaceSplit: "train",
    huggingFaceRevision: "",
    exampleId: "",
    selectedFileNames: [],
    selectedFileSignatures: [],
    capacityOverrideReason: "",
    importId: "",
    inspectionId,
    mappingPlan: null,
    mappingConfirmed: false,
    preparation: DEFAULT_PREPARATION,
    advancedRecipe: false,
    rawRecipe: "",
    datasetName: "",
    datasetDescription: "",
    datasetId: "",
    buildJobId: "",
    versionId: "",
    trainerMode: "",
    model: "",
    verifierRevisionId: "",
    trainingPlanId: "",
    trainingPlanRevisionId: "",
    modelPreparationId: "",
    capacityCheckId: "",
    proofRunId: "",
    outcomeAssessmentId: "",
    outcomeOverrideReason: "",
    fullRunId: "",
    advisorGoal: "",
    advisorModality: "",
    advisorSourceLayout: "",
    cptAdaptation: "",
    cptMaxSequenceLength: 2048,
    cptBudgetMode: "passes",
    cptTargetTokens: 1_000_000,
    cptCorpusPasses: 1,
    cptPacking: "paragraph_eos_non_overlap_v1",
  };
}

export function OwnDataStudio({ startWithExample = false, initialInspectionId = "", initialTrainingPlanRevisionId = "", initialSourcePath = "", initialScenarioRevisionId = "", initialRepairRevisionId = "" }: { startWithExample?: boolean; initialInspectionId?: string; initialTrainingPlanRevisionId?: string; initialSourcePath?: string; initialScenarioRevisionId?: string; initialRepairRevisionId?: string }) {
  const queryClient = useQueryClient();
  const [draft, setDraft] = useState<OwnDataDraft>(() => ({
    ...defaultDraft(startWithExample, initialInspectionId),
    ...(initialSourcePath ? {
      step: "source" as const,
      sourceMode: "workstation" as const,
      sourcePath: initialSourcePath,
      scenarioRevisionId: initialScenarioRevisionId,
      repairRevisionId: initialRepairRevisionId,
    } : {}),
  }));
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [inlineInspection, setInlineInspection] = useState<DatasetSourceInspection | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [helpMode, setHelpMode] = useState(false);
  const [sourceError, setSourceError] = useState<string | null>(null);
  const scenariosQuery = useTrainingScenarios({ includeUnavailable: true });
  const capabilityQuery = useInterfaceCapabilities();
  const backend = useBackendInfo();
  const inspectionQuery = useDatasetInspection(draft.inspectionId);
  const inspection = inspectionQuery.data ?? inlineInspection;
  const restoredTrainingPlan = useQuery({
    queryKey: ["training-plan-revisions", initialTrainingPlanRevisionId, "restore"],
    queryFn: () => api.trainingPlanRevision(initialTrainingPlanRevisionId),
    enabled: Boolean(initialTrainingPlanRevisionId),
    retry: false,
  });
  const scenarios = useMemo(
    () => withCorpusScenarioFallback(scenariosQuery.data?.items ?? []),
    [scenariosQuery.data?.items],
  );
  const guidedScenarios = scenarios.filter(isGuidedScenario);
  const selectedScenario = findScenario(scenarios, draft.scenarioRevisionId || draft.scenarioId);
  const selectedCandidate = inspection?.schema_candidates.find(
    (candidate) => candidate.scenario_revision_id === draft.candidateScenarioRevisionId,
  );
  const activeIndex = STEPS.findIndex((step) => step.id === draft.step);

  const examplesQuery = useQuery({
    queryKey: ["training-scenarios", selectedScenario?.id, "examples"],
    queryFn: () => api.trainingScenarioExamples(selectedScenario!.id),
    enabled: Boolean(selectedScenario),
    staleTime: 10 * 60 * 1000,
  });
  const guidedExamplesQuery = useQuery({
    queryKey: ["training-scenario-examples", "guided-gallery"],
    queryFn: api.guidedTrainingExamples,
    retry: false,
    staleTime: 10 * 60 * 1000,
  });
  const guidedExamples = guidedExamplesQuery.data?.items ?? fallbackGuidedExamples(guidedScenarios);
  const selectedScenarioExamples = examplesQuery.data?.items
    ?? examplesFromGallery(guidedExamples, selectedScenario?.revision_id);
  const advisorMutation = useMutation({
    mutationFn: async (request: ScenarioAdviceRequest) => {
      try {
        return await api.adviseTrainingScenario(request);
      } catch {
        return localScenarioAdvice(request, guidedScenarios);
      }
    },
  });

  const draftState = useWorkspaceDraft({
    surface: "own-data-studio",
    draftKey: "new-dataset",
    name: draft.datasetName || selectedScenario?.label || "Train on your data",
    value: draft,
    onRestore: (value) => {
      const fallback = defaultDraft();
      setDraft({
        ...fallback,
        ...value,
        preparation: { ...fallback.preparation, ...(value.preparation ?? {}) },
        selectedFileNames: value.selectedFileNames ?? [],
        selectedFileSignatures: value.selectedFileSignatures ?? [],
      });
      setSelectedFiles([]);
    },
  });

  useEffect(() => {
    if (!initialInspectionId || draft.inspectionId === initialInspectionId) return;
    setInlineInspection(null);
    setDraft(defaultDraft(false, initialInspectionId));
  }, [draft.inspectionId, initialInspectionId]);

  useEffect(() => {
    const revision = restoredTrainingPlan.data;
    if (!revision || draft.trainingPlanRevisionId === revision.id) return;
    setDraft((current) => ({
      ...current,
      step: "train",
      versionId: revision.dataset_version_id,
      scenarioRevisionId: revision.scenario_revision_id || current.scenarioRevisionId,
      trainerMode: revision.trainer_mode as TrainingMode,
      model: revision.model_id,
      trainingPlanId: revision.plan_id,
      trainingPlanRevisionId: revision.id,
    }));
  }, [draft.trainingPlanRevisionId, restoredTrainingPlan.data]);

  useEffect(() => {
    if (!startWithExample || draft.scenarioRevisionId || !guidedScenarios.length) return;
    const scenario = guidedScenarios.find((item) => item.canonical_shape === "sft") ?? guidedScenarios[0];
    setDraft((current) => ({
      ...current,
      scenarioId: scenario.id,
      scenarioRevisionId: scenario.revision_id,
      sourceMode: "example",
      datasetName: `${scenario.label} example`,
      trainerMode: preferredTrainerMode(scenario),
      preparation: preparationDefaults(scenario),
    }));
  }, [draft.scenarioRevisionId, guidedScenarios, startWithExample]);

  useEffect(() => {
    if (!initialSourcePath || !selectedScenario || draft.trainerMode) return;
    setDraft((current) => ({
      ...current,
      scenarioId: selectedScenario.id,
      trainerMode: preferredTrainerMode(selectedScenario),
      preparation: preparationDefaults(selectedScenario),
      datasetName: current.datasetName || `${selectedScenario.label} repaired data`,
    }));
  }, [draft.trainerMode, initialSourcePath, selectedScenario]);

  useEffect(() => {
    if (!inspection || draft.candidateScenarioRevisionId) return;
    const eligible = inspection.schema_candidates.filter((candidate) => ["high", "medium"].includes(candidate.confidence));
    const high = eligible.filter((candidate) => candidate.confidence === "high");
    const preferred = high.length === 1
      ? high[0]
      : eligible.length === 1 && eligible[0].confidence === "medium"
        ? eligible[0]
        : undefined;
    if (!preferred) return;
    const scenario = findScenario(scenarios, preferred.scenario_revision_id || preferred.scenario_id);
    setDraft((current) => ({
      ...current,
      scenarioId: scenario?.id || current.scenarioId,
      scenarioRevisionId: scenario?.revision_id || current.scenarioRevisionId,
      candidateScenarioRevisionId: preferred.scenario_revision_id,
      mappingPlan: buildSuggestedMapping(preferred, scenario, inspection),
      mappingConfirmed: false,
      trainerMode: scenario ? preferredTrainerMode(scenario) : current.trainerMode,
      preparation: scenario ? preparationDefaults(scenario) : current.preparation,
    }));
  }, [draft.candidateScenarioRevisionId, draft.scenarioRevisionId, inspection, scenarios]);

  const mappingPreviewQuery = useQuery({
    queryKey: ["dataset-inspections", draft.inspectionId, "mapping-preview", draft.mappingPlan],
    queryFn: () => api.previewDatasetMapping(draft.inspectionId, { mapping_plan: draft.mappingPlan! }),
    enabled: Boolean(draft.inspectionId && draft.mappingPlan && Object.keys(draft.mappingPlan.mappings).length),
    retry: false,
  });
  const localMappingPreview = useMemo(
    () => inspection && draft.mappingPlan ? buildLocalMappingPreview(inspection, draft.mappingPlan) : null,
    [draft.mappingPlan, inspection],
  );
  const mappingPreview = mappingPreviewQuery.data ?? localMappingPreview;
  const semanticPreviewQuery = useQuery({
    queryKey: ["dataset-inspections", draft.inspectionId, "semantic-preview", draft.mappingPlan],
    queryFn: () => api.previewDatasetSemantics(draft.inspectionId, { mapping_plan: draft.mappingPlan! }, 20),
    enabled: Boolean(draft.inspectionId && draft.mappingPlan && Object.keys(draft.mappingPlan.mappings).length),
    retry: false,
  });
  const semanticPreview = semanticPreviewQuery.data
    ?? buildLocalSemanticPreview(mappingPreview, selectedScenario);

  const recipe = useMemo(() => resolvedRecipe(draft, selectedScenario), [draft, selectedScenario]);
  const recipeError = useMemo(() => {
    if (!draft.advancedRecipe) return null;
    try {
      parseRecipeText(draft.rawRecipe);
      return null;
    } catch (error) {
      return error instanceof Error ? error.message : "Recipe YAML or JSON is invalid.";
    }
  }, [draft.advancedRecipe, draft.rawRecipe]);
  const preparationPlan = useMemo<DatasetPreparationPlan | null>(() => {
    if (!draft.mappingPlan || !selectedScenario || recipeError) return null;
    return {
      scenario_revision_id: selectedScenario.revision_id,
      mapping_plan: draft.mappingPlan,
      recipe,
      sampled: true,
    };
  }, [draft.mappingPlan, recipe, recipeError, selectedScenario]);

  const preparationQuery = useQuery({
    queryKey: ["dataset-inspections", draft.inspectionId, "preparation-preview", preparationPlan],
    queryFn: () => api.previewDatasetPreparation(draft.inspectionId, { preparation_plan: preparationPlan! }),
    enabled: Boolean(draft.inspectionId && preparationPlan && ["prepare", "version"].includes(draft.step)),
    retry: false,
  });
  const localInspectionReadiness = useMemo(
    () => buildLocalInspectionReadiness(inspection, mappingPreview, preparationPlan, selectedScenario),
    [inspection, mappingPreview, preparationPlan, selectedScenario],
  );
  const inspectionReadinessQuery = useQuery({
    queryKey: ["dataset-inspections", draft.inspectionId, "readiness", preparationPlan],
    queryFn: () => api.datasetInspectionReadiness(draft.inspectionId, { preparation_plan: preparationPlan! }),
    enabled: Boolean(draft.inspectionId && preparationPlan && ["prepare", "version"].includes(draft.step)),
    retry: false,
  });
  const inspectionReadiness = inspectionReadinessQuery.data ?? localInspectionReadiness;

  const inspectMutation = useMutation({
    mutationFn: async () => {
      setSourceError(null);
      setUploadProgress(0);
      const session = await createImportSession(draft, selectedScenario, selectedFiles);
      // Persist the durable import identity before transferring bytes. A
      // reconnect can ask the operator to reselect the same browser files and
      // continue at the server-recorded chunk boundary.
      setDraft((current) => ({ ...current, importId: session.id }));
      if (draft.sourceMode === "upload") {
        let uploaded = 0;
        const total = selectedFiles.reduce((sum, file) => sum + file.size, 0);
        for (const file of selectedFiles) {
          const record = await api.createDatasetImportFile(session.id, {
            relative_path: file.webkitRelativePath || file.name,
            size_bytes: file.size,
            content_type: file.type || "application/octet-stream",
            capacity_override_reason: draft.capacityOverrideReason.trim() || undefined,
          });
          const chunkSize = 4 * 1024 * 1024;
          const resumeAt = Math.max(0, Math.min(file.size, record.uploaded_bytes || 0));
          uploaded += resumeAt;
          setUploadProgress(total ? Math.round((uploaded / total) * 100) : 100);
          for (let start = resumeAt; start < file.size; start += chunkSize) {
            const endExclusive = Math.min(file.size, start + chunkSize);
            const content = await file.slice(start, endExclusive).arrayBuffer();
            const contentHash = await sha256Hex(content);
            await api.uploadDatasetImportFileChunk(session.id, record.id, content, {
              start,
              end: endExclusive - 1,
              total: file.size,
            }, contentHash);
            uploaded += endExclusive - start;
            setUploadProgress(total ? Math.round((uploaded / total) * 100) : 100);
          }
        }
      }
      const result = await api.inspectDatasetImport(session.id, {
        scenario_revision_id: selectedScenario?.revision_id,
      });
      return { session, result };
    },
    onSuccess: ({ session, result }) => {
      const embedded = unwrapInspection(result);
      const inspectionId = embedded?.id || result.import?.inspection_id || session.inspection_id || "";
      setInlineInspection(embedded);
      setDraft((current) => ({
        ...current,
        importId: session.id,
        inspectionId,
        step: "format",
        candidateConfirmed: false,
        mappingConfirmed: false,
      }));
      if (inspectionId) queryClient.invalidateQueries({ queryKey: ["dataset-inspections", inspectionId] });
    },
    onError: (error) => setSourceError(error instanceof Error ? error.message : "The source could not be inspected."),
  });

  const buildMutation = useMutation({
    mutationFn: async () => {
      if (!inspection || !selectedScenario || !draft.mappingPlan || !preparationPlan) {
        throw new Error("Complete inspection and mapping before publishing a version.");
      }
      let datasetId = draft.datasetId;
      if (!datasetId) {
        const registered = await api.registerInspectedDataset(inspection.id, {
          name: draft.datasetName.trim() || defaultDatasetName(selectedScenario, draft),
          description: draft.datasetDescription.trim() || undefined,
          import_id: draft.importId || undefined,
          scenario_revision_id: selectedScenario.revision_id,
          mapping_plan: draft.mappingPlan,
          preparation_plan: preparationPlan,
          capacity_override_reason: draft.capacityOverrideReason.trim() || undefined,
        });
        let dataset = unwrapDataset(registered);
        const registrationImportId = registered.import?.id || draft.importId;
        for (let attempt = 0; !dataset?.id && registrationImportId && attempt < 600; attempt += 1) {
          const session = await api.datasetImport(registrationImportId);
          if (session.published_dataset_id) {
            dataset = await api.datasetDetail(session.published_dataset_id);
            break;
          }
          if (["failed", "cancelled", "expired"].includes(session.status)) {
            throw new Error(session.error || "Dataset registration did not complete. Open Activity to retry it.");
          }
          if (registered.work_item_id) {
            const work = await api.workItem(registered.work_item_id);
            if (["failed", "cancelled", "interrupted", "needs_reconciliation"].includes(work.status)) {
              throw new Error(work.error || "Dataset registration needs attention in Activity.");
            }
          }
          await new Promise((resolve) => window.setTimeout(resolve, 500));
        }
        if (!dataset?.id) throw new Error("Dataset registration is still running. Open Activity to monitor or retry it.");
        datasetId = dataset.id;
      }
      const build = await api.buildDataset(datasetId, { recipe });
      return { datasetId, build };
    },
    onSuccess: ({ datasetId, build }) => {
      setDraft((current) => ({
        ...current,
        datasetId,
        buildJobId: build.job_id || build.id || "",
        versionId: build.version_id || current.versionId,
      }));
      queryClient.invalidateQueries({ queryKey: ["datasets"] });
    },
  });

  const buildJob = useQuery({
    queryKey: ["dataset-jobs", draft.buildJobId],
    queryFn: () => api.datasetJob(draft.buildJobId),
    enabled: Boolean(draft.buildJobId),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status && ["completed", "failed", "cancelled"].includes(status) ? false : 1_000;
    },
  });
  const builtVersions = useQuery({
    queryKey: ["datasets", draft.datasetId, "versions", "own-data-studio"],
    queryFn: () => api.datasetVersions(draft.datasetId),
    enabled: Boolean(draft.datasetId && (buildJob.data?.status === "completed" || buildJob.data?.status === "succeeded")),
  });
  const builtVersionId = draft.versionId || buildJob.data?.version_id || builtVersions.data?.items[0]?.id || "";
  const buildComplete = Boolean(builtVersionId && (!draft.buildJobId || ["completed", "succeeded"].includes(buildJob.data?.status ?? "completed")));

  useEffect(() => {
    if (!builtVersionId || draft.versionId === builtVersionId) return;
    setDraft((current) => ({ ...current, versionId: builtVersionId }));
  }, [builtVersionId, draft.versionId]);

  useEffect(() => {
    const scroller = workspaceScrollContainer();
    if (!scroller) return;
    scroller.scrollTo({ top: 0, behavior: "auto" });
  }, [draft.step]);

  function moveTo(step: StudioStep) {
    const target = STEPS.findIndex((item) => item.id === step);
    if (step === "version" && inspectionReadiness?.ready === false) return;
    if (target <= activeIndex || canEnterStep(step, draft, inspection, builtVersionId)) {
      setDraft((current) => ({ ...current, step }));
    }
  }

  function chooseScenario(scenario: TrainingScenarioDescriptor) {
    setDraft((current) => ({
      ...defaultDraft(current.sourceMode === "example"),
      sourceMode: current.sourceMode,
      scenarioId: scenario.id,
      scenarioRevisionId: scenario.revision_id,
      datasetName: current.datasetName,
      trainerMode: preferredTrainerMode(scenario),
      preparation: preparationDefaults(scenario),
    }));
  }

  function chooseExample(example: GuidedExampleDescriptor) {
    const scenario = findScenario(scenarios, example.scenario_revision_id || example.scenario_id);
    if (!scenario) return;
    setDraft((current) => ({
      ...defaultDraft(true),
      advisorGoal: current.advisorGoal,
      advisorModality: current.advisorModality,
      advisorSourceLayout: current.advisorSourceLayout,
      scenarioId: scenario.id,
      scenarioRevisionId: scenario.revision_id,
      sourceMode: "example",
      exampleId: example.id,
      datasetName: `${example.label} example`,
      trainerMode: preferredTrainerMode(scenario),
      preparation: preparationDefaults(scenario),
    }));
  }

  const context = {
    draft,
    setDraft,
    scenarios,
    guidedScenarios,
    selectedScenario,
    selectedCandidate,
    inspection,
    inspectionLoading: inspectionQuery.isLoading,
    inspectionError: inspectionQuery.error instanceof Error ? inspectionQuery.error.message : null,
    mappingPreview,
    semanticPreview,
    semanticPreviewLoading: semanticPreviewQuery.isFetching && !semanticPreviewQuery.data,
    preparationPlan: preparationQuery.data ?? preparationPlan,
    inspectionReadiness,
    inspectionReadinessLoading: inspectionReadinessQuery.isFetching && !inspectionReadinessQuery.data,
    recipe,
    recipeError,
    examples: selectedScenarioExamples,
    guidedExamples,
    advisorResult: advisorMutation.data,
    advisorPending: advisorMutation.isPending,
    adviseScenario: (request: ScenarioAdviceRequest) => advisorMutation.mutate(request),
    selectedFiles,
    setSelectedFiles,
    sourceError,
    uploadProgress,
    inspectMutation,
    buildMutation,
    buildJob: buildJob.data,
    buildComplete,
    builtVersionId,
    backendName: backend.data?.name,
  };

  return (
    <div className="min-h-full">
      {draftState.candidate && !initialTrainingPlanRevisionId && !initialSourcePath ? (
        <RestoreDraftBanner
          name={draftState.candidate.name}
          onRestore={draftState.restore}
          onDiscard={draftState.discard}
        />
      ) : null}

      <div className="grid min-h-[calc(100vh-98px)] grid-cols-1 lg:grid-cols-[230px_minmax(0,1fr)_310px]">
        <StepRail
          active={draft.step}
          draft={draft}
          inspection={inspection}
          versionId={builtVersionId}
          readiness={inspectionReadiness}
          onStep={moveTo}
        />

        <main className="min-w-0 border-y border-border-subtle lg:border-y-0 lg:border-x" aria-live="polite">
          <div key={draft.step} className="animate-[studio-in_180ms_var(--ease-out-expo)]">
            {draft.step === "goal" ? (
              <GoalStep
                scenarios={scenarios}
                guidedScenarios={guidedScenarios}
                selected={selectedScenario}
                loading={scenariosQuery.isLoading}
                error={scenariosQuery.error instanceof Error ? scenariosQuery.error.message : null}
                helpMode={helpMode}
                onHelpMode={setHelpMode}
                onSelect={chooseScenario}
                onExample={chooseExample}
                draft={draft}
                setDraft={setDraft}
                examples={guidedExamples}
                advisor={advisorMutation.data}
                advisorPending={advisorMutation.isPending}
                onAdvise={(request) => advisorMutation.mutate(request)}
              />
            ) : null}
            {draft.step === "source" ? <SourceStep {...context} /> : null}
            {draft.step === "format" ? <FormatStep {...context} /> : null}
            {draft.step === "map" ? <MapStep {...context} /> : null}
            {draft.step === "prepare" ? <PrepareStep {...context} /> : null}
            {draft.step === "version" ? <VersionStep {...context} /> : null}
            {draft.step === "train" ? <TrainStep {...context} /> : null}
          </div>

          <StudioFooter
            context={context}
            onBack={() => moveTo(STEPS[Math.max(0, activeIndex - 1)].id)}
            onNext={() => {
              const next = STEPS[Math.min(STEPS.length - 1, activeIndex + 1)].id;
              moveTo(next);
            }}
          />
        </main>

        <EvidenceInspector
          step={draft.step}
          scenario={selectedScenario}
          inspection={inspection}
          mappingPreview={mappingPreview}
          preparation={preparationQuery.data ?? preparationPlan}
          readiness={inspectionReadiness}
          draft={draft}
          buildStatus={buildJob.data?.status}
          capabilities={capabilityQuery.data?.items ?? []}
          saveState={draftState.isSaving ? "Saving" : draftState.saveError ? "Draft unavailable" : "Draft saved"}
        />
      </div>
    </div>
  );
}

type StudioContext = ReturnType<typeof buildContextType>;
// Type-only helper: its body is never called, but keeps step props in sync.
function buildContextType() {
  return {} as {
    draft: OwnDataDraft;
    setDraft: React.Dispatch<React.SetStateAction<OwnDataDraft>>;
    scenarios: TrainingScenarioDescriptor[];
    guidedScenarios: TrainingScenarioDescriptor[];
    selectedScenario?: TrainingScenarioDescriptor;
    selectedCandidate?: SchemaCandidate;
    inspection?: DatasetSourceInspection | null;
    inspectionLoading: boolean;
    inspectionError: string | null;
    mappingPreview?: MappingPreview | null;
    semanticPreview?: SemanticPreviewResponse | null;
    semanticPreviewLoading: boolean;
    preparationPlan?: DatasetPreparationPlan | null;
    inspectionReadiness?: DatasetReadiness | null;
    inspectionReadinessLoading: boolean;
    recipe: DatasetRecipe;
    recipeError: string | null;
    examples: TrainingScenarioExample[];
    guidedExamples: GuidedExampleDescriptor[];
    advisorResult?: ScenarioAdviceResult;
    advisorPending: boolean;
    adviseScenario: (request: ScenarioAdviceRequest) => void;
    selectedFiles: File[];
    setSelectedFiles: React.Dispatch<React.SetStateAction<File[]>>;
    sourceError: string | null;
    uploadProgress: number;
    inspectMutation: { isPending: boolean; isError: boolean; error: unknown; mutate: () => void };
    buildMutation: { isPending: boolean; isError: boolean; error: unknown; mutate: () => void };
    buildJob?: { status: string; stage?: string | null; progress_percent?: number | null; error?: string | null };
    buildComplete: boolean;
    builtVersionId: string;
    backendName?: string;
  };
}

function StepRail({ active, draft, inspection, versionId, readiness, onStep }: { active: StudioStep; draft: OwnDataDraft; inspection?: DatasetSourceInspection | null; versionId: string; readiness?: DatasetReadiness | null; onStep: (step: StudioStep) => void }) {
  const activeIndex = STEPS.findIndex((step) => step.id === active);
  return (
    <nav className="overflow-x-auto bg-bg-subtle/40 lg:overflow-visible" aria-label="Own data training steps">
      <ol className="flex min-w-max lg:block lg:min-w-0 lg:py-3">
        {STEPS.map((step, index) => {
          const complete = stepComplete(step.id, draft, inspection, versionId);
          const enabled = (index <= activeIndex || complete || canEnterStep(step.id, draft, inspection, versionId))
            && !(step.id === "version" && readiness?.ready === false);
          return (
            <li key={step.id} className="border-r border-border-subtle last:border-r-0 lg:border-r-0">
              <button
                type="button"
                onClick={() => enabled && onStep(step.id)}
                disabled={!enabled}
                aria-current={active === step.id ? "step" : undefined}
                className={cn(
                  "group flex min-h-16 w-44 items-start gap-3 px-4 py-3 text-left transition-colors lg:w-full",
                  active === step.id ? "bg-accent-bg/60" : enabled ? "hover:bg-surface/60" : "opacity-40",
                )}
              >
                <span className={cn(
                  "mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded-full border font-mono text-[9px]",
                  active === step.id ? "border-accent bg-accent text-accent-fg" : complete ? "border-success text-success" : "border-border-strong text-fg-subtle",
                )}>
                  {complete ? <Check className="h-3 w-3" /> : index + 1}
                </span>
                <span>
                  <span className={cn("block text-[12px] font-medium", active === step.id ? "text-accent" : "text-fg")}>{step.label}</span>
                  <span className="mt-0.5 hidden text-[10px] leading-4 text-fg-subtle lg:block">{step.detail}</span>
                </span>
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

function GoalStep({
  scenarios,
  guidedScenarios,
  selected,
  loading,
  error,
  helpMode,
  onHelpMode,
  onSelect,
  onExample,
  draft,
  setDraft,
  examples,
  advisor,
  advisorPending,
  onAdvise,
}: {
  scenarios: TrainingScenarioDescriptor[];
  guidedScenarios: TrainingScenarioDescriptor[];
  selected?: TrainingScenarioDescriptor;
  loading: boolean;
  error: string | null;
  helpMode: boolean;
  onHelpMode: (value: boolean) => void;
  onSelect: (scenario: TrainingScenarioDescriptor) => void;
  onExample: (example: GuidedExampleDescriptor) => void;
  draft: OwnDataDraft;
  setDraft: React.Dispatch<React.SetStateAction<OwnDataDraft>>;
  examples: GuidedExampleDescriptor[];
  advisor?: ScenarioAdviceResult;
  advisorPending: boolean;
  onAdvise: (request: ScenarioAdviceRequest) => void;
}) {
  const [filter, setFilter] = useState<string>("all");
  const visible = guidedScenarios.filter((scenario) => (
    filter === "all"
    || scenario.modality === filter
    || scenario.canonical_shape === filter
    || (filter === "documents" && isCorpusScenario(scenario))
  ));
  const unavailable = scenarios.filter((scenario) => !isGuidedScenario(scenario));
  const modalities = ["all", "text", "preference", "tool", "image", "audio", "documents"];
  const recommendations = advisor?.recommendations.slice(0, 3) ?? [];
  const submitAdvice = () => onAdvise({
    goal: draft.advisorGoal.trim(),
    modality: draft.advisorModality || undefined,
    source_layout: draft.advisorSourceLayout || undefined,
    include_unavailable: true,
  });
  return (
    <StepSurface
      number="01"
      title="What should the model learn?"
      detail="Choose the shape that best matches the examples you already have. Halo Forge will verify the format before anything is built."
      action={<Button variant="ghost" size="sm" onClick={() => onHelpMode(!helpMode)}><HelpCircle />{helpMode ? "Close advisor" : "Help me decide"}</Button>}
    >
      {helpMode ? (
        <div className="border-b border-border-subtle bg-bg-subtle/45">
          <div className="grid gap-4 px-5 py-5 lg:grid-cols-[minmax(0,1fr)_250px]">
            <Field label="Describe the outcome" hint="Use plain language. Advice is ranked and never selected automatically.">
              <textarea
                value={draft.advisorGoal}
                onChange={(event) => setDraft((current) => ({ ...current, advisorGoal: event.target.value }))}
                rows={3}
                placeholder="Adapt a model to our product manuals and domain language"
                className="w-full resize-y rounded-md border border-border bg-bg px-3 py-2 text-xs leading-5 text-fg outline-none transition focus:border-accent focus:ring-2 focus:ring-accent/20"
                aria-label="Describe what the model should learn"
                autoFocus
              />
            </Field>
            <div className="space-y-3">
              <Field label="Source kind">
                <Select value={draft.advisorModality || "any"} onValueChange={(value) => setDraft((current) => ({ ...current, advisorModality: value === "any" ? "" : value }))}>
                  <SelectTrigger aria-label="Advisor source kind"><SelectValue /></SelectTrigger>
                  <SelectContent><SelectItem value="any">Not sure yet</SelectItem><SelectItem value="text">Text or documents</SelectItem><SelectItem value="image">Images with text</SelectItem><SelectItem value="audio">Audio with text</SelectItem></SelectContent>
                </Select>
              </Field>
              <Field label="Current layout">
                <Select value={draft.advisorSourceLayout || "unknown"} onValueChange={(value) => setDraft((current) => ({ ...current, advisorSourceLayout: value === "unknown" ? "" : value }))}>
                  <SelectTrigger aria-label="Advisor source layout"><SelectValue /></SelectTrigger>
                  <SelectContent><SelectItem value="unknown">Not sure yet</SelectItem><SelectItem value="jsonl">Rows or records</SelectItem><SelectItem value="document_directory">Document folder</SelectItem><SelectItem value="pdf">PDF documents</SelectItem><SelectItem value="markdown">Markdown or text</SelectItem><SelectItem value="media_directory_manifest">Media folder and manifest</SelectItem></SelectContent>
                </Select>
              </Field>
            </div>
          </div>
          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border-subtle px-5 py-3">
            <p className="max-w-2xl text-[10px] leading-4 text-fg-subtle">The advisor uses only the goal and source details entered here. You still confirm the scenario after inspection.</p>
            <Button size="sm" variant="primary" disabled={advisorPending || !draft.advisorGoal.trim()} onClick={submitAdvice}>{advisorPending ? <Loader2 className="animate-spin" /> : <Sparkles />}Find the best fit</Button>
          </div>
          {recommendations.length ? (
            <div className="divide-y divide-border-subtle border-t border-border-subtle">
              {recommendations.map((recommendation, index) => {
                const scenario = findScenario(guidedScenarios, recommendation.scenario_revision_id || recommendation.scenario_id);
                if (!scenario) return null;
                return (
                  <div key={recommendation.scenario_revision_id} className="grid gap-3 bg-bg px-5 py-4 sm:grid-cols-[30px_minmax(0,1fr)_auto]">
                    <span className="font-mono text-[10px] text-accent">{String(index + 1).padStart(2, "0")}</span>
                    <div>
                      <div className="flex flex-wrap items-center gap-2"><span className="text-[13px] font-medium text-fg">{recommendation.label}</span><ConfidenceBadge confidence={recommendation.confidence} /></div>
                      <p className="mt-1 text-[11px] leading-5 text-fg-muted">{recommendation.expected_outcome || scenario.description}</p>
                      <p className="mt-2 text-[10px] leading-4 text-fg-subtle">{recommendation.why_fit.slice(0, 2).join(" ")}</p>
                      {recommendation.cautions?.length ? <p className="mt-1 text-[10px] leading-4 text-warning">{recommendation.cautions[0]}</p> : null}
                    </div>
                    <Button size="sm" variant={selected?.revision_id === scenario.revision_id ? "secondary" : "primary"} onClick={() => onSelect(scenario)}>{selected?.revision_id === scenario.revision_id ? <CheckCircle2 /> : <ArrowRight />}{selected?.revision_id === scenario.revision_id ? "Selected" : "Use this scenario"}</Button>
                  </div>
                );
              })}
              <p className="bg-bg-subtle/45 px-5 py-3 text-[10px] leading-4 text-fg-subtle">{advisor?.explanation}</p>
            </div>
          ) : null}
        </div>
      ) : null}
      <div className="flex flex-wrap items-center gap-1 border-b border-border-subtle px-5 py-2">
        {modalities.map((value) => (
          <button key={value} type="button" aria-pressed={filter === value} onClick={() => setFilter(value)} className={cn("rounded-sm px-2.5 py-1 text-[11px] capitalize", filter === value ? "bg-accent-bg text-accent" : "text-fg-muted hover:bg-surface hover:text-fg")}>{value}</button>
        ))}
      </div>
      {loading ? <LoadingState label="Loading verified scenarios" /> : error ? <ErrorState label={error} /> : (
        <div className="divide-y divide-border-subtle">
          {visible.map((scenario) => <ScenarioRow key={scenario.revision_id} scenario={scenario} selected={selected?.revision_id === scenario.revision_id} onSelect={() => onSelect(scenario)} />)}
          {!visible.length ? <EmptyState icon={Database} title="No verified scenario in this group" detail="Choose another group or inspect unavailable capabilities below." /> : null}
        </div>
      )}
      {unavailable.length ? (
        <details className="border-t border-border-subtle px-5 py-4">
          <summary className="cursor-pointer text-[11px] font-medium text-fg-muted">Unavailable or unverified scenarios ({unavailable.length})</summary>
          <div className="mt-3 space-y-2">
            {unavailable.map((scenario) => (
              <div key={scenario.revision_id} className="flex items-start justify-between gap-3 border-l border-border-strong pl-3 text-[11px]">
                <span className="text-fg-muted">{scenario.label}</span>
                <span className="max-w-[55ch] text-right text-fg-subtle">{scenario.unavailable_reason || "No verified data-to-weight-update contract is available."}</span>
              </div>
            ))}
          </div>
        </details>
      ) : null}
      <div className="border-t border-border-subtle">
        <div className="flex flex-wrap items-end justify-between gap-3 px-5 py-4"><div><div className="text-xs font-medium text-fg">Try a working example</div><div className="mt-1 text-[11px] leading-5 text-fg-muted">Start with a small verified source, inspect its semantic preview, and carry the same workflow into your own data.</div></div><Badge size="sm" tone="neutral">{examples.length} examples</Badge></div>
        <div className="grid gap-px border-t border-border-subtle bg-border-subtle md:grid-cols-2">
          {examples.map((example) => {
            const scenario = findScenario(guidedScenarios, example.scenario_revision_id || example.scenario_id);
            const active = draft.sourceMode === "example" && draft.exampleId === example.id;
            return (
              <button key={`${example.scenario_revision_id}-${example.id}`} type="button" onClick={() => onExample(example)} aria-pressed={active} className={cn("min-h-40 bg-bg px-5 py-4 text-left transition-colors hover:bg-surface/45", active && "bg-accent-bg/55")}>
                <span className="flex items-center justify-between gap-3"><span className="text-[12px] font-medium text-fg">{example.label}</span>{active ? <CheckCircle2 className="h-4 w-4 text-accent" /> : <ArrowRight className="h-4 w-4 text-fg-disabled" />}</span>
                <span className="mt-1 block text-[10px] text-fg-subtle">{scenario?.label || humanize(example.modality)} · {example.record_count} reviewed {example.record_count === 1 ? "record" : "records"}</span>
                <span className="mt-3 block text-[11px] leading-5 text-fg-muted">{example.description}</span>
                <span className="mt-3 block border-l border-border-strong pl-3 text-[10px] leading-4 text-fg-subtle">{example.expected_outcome}</span>
              </button>
            );
          })}
        </div>
      </div>
    </StepSurface>
  );
}

function ScenarioRow({ scenario, selected, onSelect }: { scenario: TrainingScenarioDescriptor; selected: boolean; onSelect: () => void }) {
  const Icon = scenarioIcon(scenario);
  return (
    <button type="button" onClick={onSelect} aria-pressed={selected} className={cn("grid w-full gap-3 px-5 py-4 text-left transition-colors sm:grid-cols-[36px_minmax(0,1fr)_auto]", selected ? "bg-accent-bg/55" : "hover:bg-surface/45")}> 
      <span className={cn("grid h-9 w-9 place-items-center rounded-md border", selected ? "border-accent text-accent" : "border-border text-fg-subtle")}><Icon className="h-4 w-4" /></span>
      <span className="min-w-0">
        <span className="flex flex-wrap items-center gap-2"><span className="text-[13px] font-medium text-fg">{scenario.label}</span><Badge size="sm" tone="neutral">{scenarioKindLabel(scenario)}</Badge></span>
        <span className="mt-1 block max-w-2xl text-[11px] leading-5 text-fg-muted">{scenario.description}</span>
        <span className="mt-2 block text-[10px] text-fg-subtle">
          {isCorpusScenario(scenario)
            ? "Bring readable documents; extraction adds identity, hashes, and source provenance."
            : `Needs ${scenarioFields(scenario, true).map((field) => field.label || field.name).join(", ")}`}
        </span>
      </span>
      <span className="flex items-center gap-2 self-center text-[11px] text-fg-subtle">{selected ? <><CheckCircle2 className="h-4 w-4 text-accent" />Selected</> : <ArrowRight className="h-4 w-4" />}</span>
    </button>
  );
}

function SourceStep(context: StudioContext) {
  const { draft, setDraft, selectedScenario, examples, selectedFiles, setSelectedFiles, sourceError, uploadProgress, inspectMutation } = context;
  const remote = connectionMode() === "remote";
  const desktop = isDesktopRuntime();
  const corpus = isCorpusScenario(selectedScenario);
  const extractors = useQuery({
    queryKey: ["document-extractors", "guided-own-data"],
    queryFn: api.documentExtractors,
    enabled: corpus,
    retry: false,
    staleTime: 10 * 60 * 1000,
  });
  const documentAccept = ".txt,.text,.md,.markdown,.mdown,.mkd,.html,.htm,.pdf,.docx,.json,.jsonl,.jl,.csv,.tsv,.parquet";
  const huggingFaceOptions = useQuery({
    queryKey: ["dataset-imports", "huggingface-options", draft.huggingFaceId, draft.huggingFaceRevision],
    queryFn: () => api.huggingFaceDatasetOptions(draft.huggingFaceId.trim(), draft.huggingFaceRevision.trim()),
    enabled: false,
    retry: false,
  });
  const selectedHuggingFaceOption = huggingFaceOptions.data?.items.find((item) => (item.config ?? "") === draft.huggingFaceConfig)
    ?? huggingFaceOptions.data?.items[0];
  const selectFiles = (files: File[]) => {
    const names = files.map((file) => file.webkitRelativePath || file.name);
    const signatures = files.map((file) => `${file.webkitRelativePath || file.name}:${file.size}:${file.lastModified}`);
    setSelectedFiles(files);
    setDraft((current) => ({
      ...current,
      selectedFileNames: names,
      selectedFileSignatures: signatures,
      importId: sameStrings(current.selectedFileSignatures, signatures) ? current.importId : "",
      inspectionId: sameStrings(current.selectedFileSignatures, signatures) ? current.inspectionId : "",
    }));
  };
  const onDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    selectFiles(Array.from(event.dataTransfer.files));
  };
  async function chooseNative(kind: "file" | "folder") {
    try {
      const result = await pickDatasetSource({ kind, multiple: kind === "file" });
      if (!result?.paths.length) return;
      setDraft((current) => ({ ...current, sourceMode: "desktop", sourcePath: result.paths[0], selectedFileNames: result.paths }));
    } catch (error) {
      // The mutation's visible error zone also serves native bridge failures.
      setDraft((current) => ({ ...current, sourceMode: "desktop", sourcePath: "" }));
      console.error("Native dataset picker failed", error);
    }
  }
  async function browseHuggingFace() {
    const result = await huggingFaceOptions.refetch();
    const first = result.data?.items[0];
    if (!first) return;
    setDraft((current) => {
      const currentOption = result.data?.items.find((item) => (item.config ?? "") === current.huggingFaceConfig);
      const option = currentOption ?? first;
      return {
        ...current,
        huggingFaceConfig: option.config ?? "",
        huggingFaceSplit: option.splits.includes(current.huggingFaceSplit) ? current.huggingFaceSplit : option.splits[0] ?? "train",
      };
    });
  }
  return (
    <StepSurface
      number="02"
      title={corpus ? "Choose the document corpus" : "Choose the source"}
      detail={corpus
        ? "Add documents as files, a folder, a pinned dataset, or a verified example. Extraction preserves source provenance before preparation."
        : "Workstation paths stay referenced and hashed. Browser uploads are copied into managed local storage after checksum verification."}
    >
      <div className="grid gap-px border-b border-border-subtle bg-border-subtle sm:grid-cols-2 xl:grid-cols-4">
        {desktop ? <SourceModeButton icon={FolderOpen} label="Desktop picker" detail="Reference a file or folder" active={draft.sourceMode === "desktop"} onClick={() => setDraft((current) => ({ ...current, sourceMode: "desktop" }))} /> : null}
        <SourceModeButton icon={UploadCloud} label={remote ? "Upload from this device" : "Browser upload"} detail="Managed local copy" active={draft.sourceMode === "upload"} onClick={() => setDraft((current) => ({ ...current, sourceMode: "upload" }))} />
        <SourceModeButton icon={Database} label={remote ? "Path on the workstation" : "Workstation path"} detail="Reference without copying" active={draft.sourceMode === "workstation"} onClick={() => setDraft((current) => ({ ...current, sourceMode: "workstation" }))} />
        <SourceModeButton icon={Cloud} label="Hugging Face" detail="Pin repository and revision" active={draft.sourceMode === "huggingface"} onClick={() => setDraft((current) => ({ ...current, sourceMode: "huggingface" }))} />
        {draft.sourceMode === "example" ? <SourceModeButton icon={Sparkles} label="Working example" detail="Small verified fixture" active onClick={() => undefined} /> : null}
      </div>
      {corpus ? <CorpusSourceFlow extractors={extractors.data?.items} loading={extractors.isLoading} /> : null}

      <div className="px-5 py-5">
        {draft.sourceMode === "desktop" ? (
          <div className="space-y-3">
            <div className="flex flex-wrap gap-2"><Button type="button" variant="secondary" onClick={() => chooseNative("file")}><FileJson />Choose file</Button><Button type="button" variant="secondary" onClick={() => chooseNative("folder")}><FolderOpen />Choose folder</Button></div>
            {draft.sourcePath ? <SelectedSource value={draft.sourcePath} detail="Referenced on this workstation; the source will not be modified." /> : <Hint>Select JSON, JSONL, CSV, TSV, Parquet, or a media folder with a supported manifest or sidecars.</Hint>}
          </div>
        ) : null}
        {draft.sourceMode === "upload" ? (
          <div className="space-y-3">
            <label onDragOver={(event) => event.preventDefault()} onDrop={onDrop} className="flex min-h-40 cursor-pointer flex-col items-center justify-center border border-dashed border-border-strong bg-bg-subtle/35 px-6 py-8 text-center transition-colors hover:border-accent hover:bg-accent-bg/20 focus-within:border-accent">
              <UploadCloud className="h-7 w-7 text-fg-disabled" />
              <span className="mt-3 text-xs font-medium text-fg">Drop files or a folder here</span>
              <span className="mt-1 max-w-md text-[11px] leading-5 text-fg-muted">Relative folder paths are preserved. Large files upload in resumable 4 MB chunks.</span>
              <input type="file" multiple accept={corpus ? documentAccept : undefined} className="sr-only" onChange={(event) => selectFiles(Array.from(event.target.files ?? []))} />
            </label>
            <div className="flex flex-wrap items-center gap-2">
              <label className="inline-flex h-8 cursor-pointer items-center gap-2 rounded-md border border-border bg-surface px-3 text-xs font-medium text-fg hover:bg-surface-hover"><FileJson className="h-4 w-4" />Choose files<input type="file" multiple accept={corpus ? documentAccept : undefined} className="sr-only" onChange={(event) => selectFiles(Array.from(event.target.files ?? []))} /></label>
              <label className="inline-flex h-8 cursor-pointer items-center gap-2 rounded-md border border-border bg-surface px-3 text-xs font-medium text-fg hover:bg-surface-hover"><FolderOpen className="h-4 w-4" />Choose folder<input type="file" multiple className="sr-only" {...({ webkitdirectory: "", directory: "" } as Record<string, string>)} onChange={(event) => selectFiles(Array.from(event.target.files ?? []))} /></label>
              <span className="text-[10px] text-fg-subtle">{selectedFiles.length ? `${selectedFiles.length} selected · ${formatBytes(selectedFiles.reduce((sum, file) => sum + file.size, 0))}` : "Nothing selected"}</span>
            </div>
            {selectedFiles.length ? <FileList files={selectedFiles} /> : draft.selectedFileNames.length ? <Hint>Reselect {draft.selectedFileNames.length} uploaded file{draft.selectedFileNames.length === 1 ? "" : "s"} after restoring this draft; browser files are never retained without permission.</Hint> : null}
          </div>
        ) : null}
        {draft.sourceMode === "workstation" ? (
          <Field label={remote ? "Path on the workstation running Halo Forge" : "Workstation file or folder path"} hint={remote ? "This is not a path on the device viewing the browser." : "The source stays where it is and is referenced by a verified content hash."}>
            <Input value={draft.sourcePath} onChange={(event) => setDraft((current) => ({ ...current, sourcePath: event.target.value }))} placeholder="/data/project/train.jsonl" mono autoFocus />
          </Field>
        ) : null}
        {draft.sourceMode === "huggingface" ? (
          <div className="space-y-4">
            <Field label="Dataset repository" hint="Use an organization/name dataset ID."><Input value={draft.huggingFaceId} onChange={(event) => setDraft((current) => ({ ...current, huggingFaceId: event.target.value }))} placeholder="organization/dataset" mono /></Field>
            <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_auto] md:items-end">
              <Field label="Pinned revision" hint="A branch name resolves to an immutable commit before registration."><Input value={draft.huggingFaceRevision} onChange={(event) => setDraft((current) => ({ ...current, huggingFaceRevision: event.target.value }))} placeholder="commit SHA or tag" mono /></Field>
              <Button type="button" variant="secondary" disabled={!draft.huggingFaceId.trim() || !draft.huggingFaceRevision.trim() || huggingFaceOptions.isFetching} onClick={browseHuggingFace}>{huggingFaceOptions.isFetching ? <Loader2 className="animate-spin" /> : <Cloud />}Browse configs and splits</Button>
            </div>
            {huggingFaceOptions.isError ? <ErrorBanner title="Hugging Face metadata could not be loaded" detail={(huggingFaceOptions.error as Error).message} tone="warning" /> : null}
            {huggingFaceOptions.data ? <div className="space-y-3 border-l-2 border-accent bg-accent-bg/20 px-4 py-3"><div className="text-[10px] text-fg-muted">Pinned to <span className="font-mono text-fg">{shortHash(huggingFaceOptions.data.resolved_revision)}</span></div><div className="grid gap-3 sm:grid-cols-2"><Field label="Config"><Select value={draft.huggingFaceConfig || "__default__"} onValueChange={(value) => { const config = value === "__default__" ? "" : value; const option = huggingFaceOptions.data?.items.find((item) => (item.config ?? "") === config); setDraft((current) => ({ ...current, huggingFaceConfig: config, huggingFaceSplit: option?.splits.includes(current.huggingFaceSplit) ? current.huggingFaceSplit : option?.splits[0] ?? "train" })); }}><SelectTrigger aria-label="Hugging Face dataset config"><SelectValue /></SelectTrigger><SelectContent>{huggingFaceOptions.data.items.map((item) => <SelectItem key={item.config ?? "__default__"} value={item.config ?? "__default__"}>{item.config || "Default"}</SelectItem>)}</SelectContent></Select></Field><Field label="Split"><Select value={draft.huggingFaceSplit} onValueChange={(value) => setDraft((current) => ({ ...current, huggingFaceSplit: value }))}><SelectTrigger aria-label="Hugging Face dataset split"><SelectValue placeholder="Choose a split" /></SelectTrigger><SelectContent>{(selectedHuggingFaceOption?.splits ?? []).map((split) => <SelectItem key={split} value={split}>{split}</SelectItem>)}</SelectContent></Select></Field></div></div> : <Hint>Enter a repository and revision, then browse the available configs and splits before inspection.</Hint>}
          </div>
        ) : null}
        {draft.sourceMode === "example" ? (
          <div className="space-y-3">
            <Field label="Verified example"><Select value={draft.exampleId || examples[0]?.id || ""} onValueChange={(value) => setDraft((current) => ({ ...current, exampleId: value }))}><SelectTrigger aria-label="Choose a working example"><SelectValue placeholder="Choose an example" /></SelectTrigger><SelectContent>{examples.map((example) => <SelectItem key={example.id} value={example.id}>{example.label}</SelectItem>)}</SelectContent></Select></Field>
            <Hint>{examples.find((example) => example.id === (draft.exampleId || examples[0]?.id))?.description || `A small ${selectedScenario?.label || "training"} fixture will be copied into managed source storage.`}</Hint>
          </div>
        ) : null}
        {sourceError ? <ErrorBanner title="Source could not be inspected" detail={sourceError} /> : null}
        {sourceError && /disk|capacity|reserve/i.test(sourceError) ? (
          <div className="mt-4">
            <Field label="Reviewed disk-capacity override reason" hint="Only continue if you have reviewed the forecast. The reason is retained with the import and remains visible in its history.">
              <Input value={draft.capacityOverrideReason} onChange={(event) => setDraft((current) => ({ ...current, capacityOverrideReason: event.target.value }))} placeholder="Why is it acceptable to cross the reserve?" />
            </Field>
          </div>
        ) : null}
        {inspectMutation.isPending ? <ProgressStrip label={uploadProgress && uploadProgress < 100 ? "Uploading source" : "Inspecting every record"} progress={uploadProgress && uploadProgress < 100 ? uploadProgress : undefined} /> : null}
      </div>
    </StepSurface>
  );
}

function FormatStep(context: StudioContext) {
  const { draft, setDraft, inspection, inspectionLoading, inspectionError, scenarios, selectedScenario } = context;
  if (!inspection && inspectionError) return <StepSurface number="03" title="Inspection could not be restored" detail="The linked source is unchanged. Return to Source or retry from Activity."><ErrorState label={inspectionError} /></StepSurface>;
  if (!inspection || ["queued", "running"].includes(inspection.status)) {
    return <StepSurface number="03" title="Inspecting the source" detail="Halo Forge is streaming the complete source for counts and field coverage, then retaining a bounded deterministic preview."><LoadingState label={inspection?.stage || (inspectionLoading ? "Restoring inspection" : "Waiting for the inspection worker")} progress={inspection?.progress_percent} /></StepSurface>;
  }
  if (inspection.status === "failed") return <StepSurface number="03" title="Inspection needs attention" detail="The source is unchanged. Fix the problem below, then retry."><ErrorState label={inspection.error || "Inspection failed."} /></StepSurface>;
  const completedInspection: DatasetSourceInspection = inspection;
  const corpus = isCorpusScenario(selectedScenario);
  const extraction = extractionSummary(inspection);
  const candidates = inspection.schema_candidates ?? [];
  const manualScenarios = scenarios.filter(isGuidedScenario).sort((left, right) => {
    if (left.revision_id === draft.scenarioRevisionId) return -1;
    if (right.revision_id === draft.scenarioRevisionId) return 1;
    return left.label.localeCompare(right.label);
  });
  function chooseManualScenario(revisionId: string) {
    const scenario = findScenario(scenarios, revisionId);
    if (!scenario) return;
    const manualCandidate: SchemaCandidate = {
      scenario_id: scenario.id,
      scenario_revision_id: scenario.revision_id,
      label: scenario.label,
      confidence: "manual",
      coverage: 0,
      reasons: ["Selected manually after reviewing the complete-source inspection."],
      suggested_mapping: {},
      missing_fields: [],
    };
    setDraft((current) => ({
      ...current,
      scenarioId: scenario.id,
      scenarioRevisionId: scenario.revision_id,
      candidateScenarioRevisionId: scenario.revision_id,
      candidateConfirmed: false,
      mappingPlan: buildSuggestedMapping(manualCandidate, scenario, completedInspection),
      mappingConfirmed: false,
      trainerMode: preferredTrainerMode(scenario),
      preparation: preparationDefaults(scenario),
    }));
  }
  return (
    <StepSurface
      number="03"
      title={corpus ? "Review document extraction" : "Confirm the detected format"}
      detail={corpus
        ? `${formatInteger(extraction.documentCount ?? inspection.row_count)} documents were inspected. Review extraction outcomes before confirming the corpus shape.`
        : `${formatInteger(inspection.row_count)} records were scanned. Detection uses the retained sample, never one convenient row.`}
    >
      <div className="grid gap-px border-b border-border-subtle bg-border-subtle sm:grid-cols-2 lg:grid-cols-5">
        <Readout label={corpus ? "Documents found" : "Records scanned"} value={formatInteger(extraction.documentCount ?? inspection.row_count)} />
        <Readout label={corpus ? "Extracted" : "Valid records"} value={formatInteger(corpus ? extraction.extracted ?? inspection.valid_records : inspection.valid_records)} />
        <Readout label={corpus ? "Quarantined" : "Invalid records"} value={formatInteger(corpus ? extraction.quarantined ?? inspection.invalid_records : inspection.invalid_records)} />
        <Readout label={corpus ? "Source fields" : "Fields"} value={String(inspection.fields.length)} />
        <Readout label="Preview retained" value={formatInteger(inspection.sample_count)} />
      </div>
      {corpus ? <CorpusExtractionReview inspection={inspection} /> : null}
      {inspection.parse_errors?.length || (inspection.invalid_records ?? 0) > 0 ? <div className="flex flex-wrap items-center justify-between gap-3 border-b border-warning/25 bg-warning/5 px-5 py-3"><ErrorBanner title={`${inspection.parse_errors?.length ?? inspection.invalid_records ?? 0} data issue${(inspection.parse_errors?.length ?? inspection.invalid_records ?? 0) === 1 ? "" : "s"}`} detail="Review deterministic fixes without editing the original source." tone="warning" /><Button asChild variant="secondary" size="sm"><Link to="/datasets/repair" search={{ inspection: inspection.id, session: undefined, source: undefined }}><Wrench />Fix data</Link></Button></div> : null}
      <div className="divide-y divide-border-subtle">
        {candidates.map((candidate) => {
          const scenario = findScenario(scenarios, candidate.scenario_revision_id || candidate.scenario_id);
          const active = draft.candidateScenarioRevisionId === candidate.scenario_revision_id;
          return <button key={candidate.scenario_revision_id} type="button" onClick={() => setDraft((current) => ({ ...current, candidateScenarioRevisionId: candidate.scenario_revision_id, scenarioId: scenario?.id || candidate.scenario_id, scenarioRevisionId: candidate.scenario_revision_id, candidateConfirmed: false, mappingPlan: buildSuggestedMapping(candidate, scenario, inspection), mappingConfirmed: false, trainerMode: scenario ? preferredTrainerMode(scenario) : current.trainerMode, preparation: scenario ? preparationDefaults(scenario) : current.preparation }))} className={cn("grid w-full gap-3 px-5 py-4 text-left sm:grid-cols-[minmax(0,1fr)_110px]", active ? "bg-accent-bg/55" : "hover:bg-surface/45")}><span><span className="flex flex-wrap items-center gap-2 text-[13px] font-medium text-fg">{scenario?.label || candidate.label || "Detected training shape"}<ConfidenceBadge confidence={candidate.confidence} /></span><span className="mt-1 block text-[11px] leading-5 text-fg-muted">{candidate.reasons?.join(" ") || `${percent(candidate.coverage)} of required fields are present in the retained inspection sample.`}</span>{candidate.missing_fields?.length ? <span className="mt-1 block text-[10px] text-warning">Needs attention: {candidate.missing_fields.map(humanize).join(", ")}</span> : null}</span><span className="self-center text-right font-mono text-[11px] text-fg-subtle">{percent(candidate.coverage)}</span></button>;
        })}
        {!candidates.length ? <EmptyState icon={CircleDashed} title="No safe format match" detail="Choose the scenario you intended below or return to Source. Halo Forge will not preselect one when required coverage is below the safe threshold." /> : null}
      </div>
      <div className="grid gap-3 border-t border-border-subtle bg-bg-subtle/35 px-5 py-4 sm:grid-cols-[minmax(0,1fr)_300px] sm:items-end">
        <div><div className="text-xs font-medium text-fg">Choose a scenario manually</div><div className="mt-1 text-[11px] leading-5 text-fg-muted">Use this when the detector is ambiguous or your field names are unusual. The next step will leave unmatched required fields visibly unconnected.</div></div>
        <Field label="Verified scenario"><Select value={draft.candidateScenarioRevisionId || ""} onValueChange={chooseManualScenario}><SelectTrigger aria-label="Choose a scenario manually"><SelectValue placeholder="Choose the intended format" /></SelectTrigger><SelectContent>{manualScenarios.map((scenario) => <SelectItem key={scenario.revision_id} value={scenario.revision_id}>{scenario.label}</SelectItem>)}</SelectContent></Select></Field>
      </div>
      {draft.candidateScenarioRevisionId ? <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border-subtle bg-bg-subtle/45 px-5 py-4"><div><div className="text-xs font-medium text-fg">{corpus ? "Confirm extraction and corpus shape" : "Confirm this interpretation"}</div><div className="mt-1 text-[11px] text-fg-muted">You must confirm even a high-confidence match before mapping fields.</div></div><Button variant={draft.candidateConfirmed ? "secondary" : "primary"} onClick={() => setDraft((current) => ({ ...current, candidateConfirmed: true }))}>{draft.candidateConfirmed ? <CheckCircle2 /> : <ShieldCheck />}{draft.candidateConfirmed ? "Format confirmed" : `Confirm ${findScenario(scenarios, draft.candidateScenarioRevisionId)?.label || selectedScenario?.label || "format"}`}</Button></div> : null}
    </StepSurface>
  );
}

function MapStep(context: StudioContext) {
  const { draft, setDraft, selectedScenario, inspection, mappingPreview, semanticPreview, semanticPreviewLoading } = context;
  if (!selectedScenario || !inspection || !draft.mappingPlan) return <StepSurface number="04" title="Map fields" detail="Confirm a source format first."><EmptyState icon={Braces} title="No mapping plan yet" detail="Return to Format and confirm the scenario that matches your data." /></StepSurface>;
  const fields = [...scenarioFields(selectedScenario, true), ...scenarioFields(selectedScenario, false)];
  const mediaRoot = inspection.preview_records.find((record) => typeof record._media_root === "string")?._media_root as string | undefined;
  const requiredMapped = scenarioFields(selectedScenario, true).every((field) => mappingExpressionReady(draft.mappingPlan!.mappings[field.name]));
  function updateMapping(target: string, expression?: FieldMappingExpression) {
    setDraft((current) => {
      if (!current.mappingPlan) return current;
      const mappings = { ...current.mappingPlan.mappings };
      if (expression) mappings[target] = expression;
      else delete mappings[target];
      return { ...current, mappingConfirmed: false, mappingPlan: { ...current.mappingPlan, mappings, confirmed: false } };
    });
  }
  return (
    <StepSurface number="04" title="Connect source fields to training" detail="Each canonical field must point to source data or an explicit transform. The preview updates without changing the source.">
      <div className="grid gap-px bg-border-subtle lg:grid-cols-[minmax(0,1.05fr)_minmax(270px,.75fr)]">
        <div className="bg-bg">
          <div className="grid grid-cols-[110px_minmax(0,1fr)] border-b border-border-subtle bg-bg-subtle px-5 py-2 text-[9.5px] font-medium uppercase tracking-[0.12em] text-fg-disabled"><span>Training field</span><span>Source expression</span></div>
          <div className="divide-y divide-border-subtle">
            {fields.map((field) => <MappingRow key={field.name} field={field} expression={draft.mappingPlan!.mappings[field.name]} sourceFields={inspection.fields.map((item) => item.name)} mediaRoot={mediaRoot} onChange={(expression) => updateMapping(field.name, expression)} />)}
          </div>
        </div>
        <MappingPreviewPane preview={mappingPreview} semantic={semanticPreview} scenario={selectedScenario} loading={semanticPreviewLoading} />
      </div>
      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border-subtle bg-bg-subtle/45 px-5 py-4"><div><div className="text-xs font-medium text-fg">{requiredMapped ? "Required fields are connected" : "Complete the required mappings"}</div><div className="mt-1 text-[11px] text-fg-muted">{mappingPreview ? `${mappingPreview.valid_count} of ${mappingPreview.total_sampled} preview records validate.` : "Preview is updating."}</div></div><Button variant={draft.mappingConfirmed ? "secondary" : "primary"} disabled={!requiredMapped || mappingPreview?.ready === false} onClick={() => setDraft((current) => ({ ...current, mappingConfirmed: true, mappingPlan: current.mappingPlan ? { ...current.mappingPlan, confirmed: true } : null }))}>{draft.mappingConfirmed ? <CheckCircle2 /> : <ShieldCheck />}{draft.mappingConfirmed ? "Mapping confirmed" : "Confirm mapping"}</Button></div>
    </StepSurface>
  );
}

function MappingRow({ field, expression, sourceFields, mediaRoot, onChange }: { field: TrainingScenarioField; expression?: FieldMappingExpression; sourceFields: string[]; mediaRoot?: string; onChange: (expression?: FieldMappingExpression) => void }) {
  const kind: FieldMappingExpression["kind"] | "none" = expression?.kind ?? (field.required ? "direct" : "none");
  return (
    <div className="grid gap-3 px-5 py-3 sm:grid-cols-[110px_minmax(0,1fr)]">
      <div><div className="flex items-center gap-1.5 text-[12px] font-medium text-fg">{field.label || field.name}{field.required ? <span className="text-accent">*</span> : null}</div><div className="mt-1 text-[10px] leading-4 text-fg-subtle">{field.description || field.value_type || "Canonical value"}</div></div>
      <div className="space-y-2">
        <div className="grid gap-2 sm:grid-cols-[125px_minmax(0,1fr)]">
          <Select value={kind} onValueChange={(value) => onChange(value === "none" ? undefined : defaultExpression(value as FieldMappingExpression["kind"], sourceFields[0] || "", mediaRoot))}><SelectTrigger aria-label={`Mapping type for ${field.label || field.name}`}><SelectValue /></SelectTrigger><SelectContent>{!field.required ? <SelectItem value="none">Not included</SelectItem> : null}<SelectItem value="direct">Source field</SelectItem><SelectItem value="constant">Constant</SelectItem><SelectItem value="concat">Combine fields</SelectItem><SelectItem value="nested_path">Nested path</SelectItem><SelectItem value="conversation">Conversation roles</SelectItem><SelectItem value="media_root">Media path</SelectItem></SelectContent></Select>
          {kind === "direct" ? <Select value={expression?.kind === "direct" ? expression.source : ""} onValueChange={(source) => onChange({ kind: "direct", source })}><SelectTrigger aria-label={`Source field for ${field.label || field.name}`}><SelectValue placeholder="Choose source field" /></SelectTrigger><SelectContent>{sourceFields.map((source) => <SelectItem key={source} value={source}>{source}</SelectItem>)}</SelectContent></Select> : null}
          {kind === "constant" ? <Input value={expression?.kind === "constant" ? String(expression.value ?? "") : ""} onChange={(event) => onChange({ kind: "constant", value: event.target.value })} placeholder="Constant value" aria-label={`Constant for ${field.label || field.name}`} /> : null}
          {kind === "concat" ? <Input value={expression?.kind === "concat" ? expression.sources.join(", ") : ""} onChange={(event) => onChange({ kind: "concat", sources: event.target.value.split(",").map((value) => value.trim()).filter(Boolean), separator: "\n" })} placeholder="field_a, field_b" aria-label={`Combined fields for ${field.label || field.name}`} /> : null}
          {kind === "nested_path" ? <div className="grid grid-cols-2 gap-2"><Select value={expression?.kind === "nested_path" ? expression.source : ""} onValueChange={(source) => onChange({ kind: "nested_path", source, path: expression?.kind === "nested_path" ? expression.path : "" })}><SelectTrigger aria-label={`Nested source for ${field.label || field.name}`}><SelectValue placeholder="Source" /></SelectTrigger><SelectContent>{sourceFields.map((source) => <SelectItem key={source} value={source}>{source}</SelectItem>)}</SelectContent></Select><Input value={expression?.kind === "nested_path" ? expression.path : ""} onChange={(event) => onChange({ kind: "nested_path", source: expression?.kind === "nested_path" ? expression.source : sourceFields[0] || "", path: event.target.value })} placeholder="items.0.text" aria-label={`Nested path for ${field.label || field.name}`} /></div> : null}
          {kind === "conversation" ? <Select value={expression?.kind === "conversation" ? expression.source : ""} onValueChange={(source) => onChange({ kind: "conversation", source, role_field: "role", content_field: "content", role_map: { human: "user", assistant: "assistant", system: "system" } })}><SelectTrigger aria-label={`Conversation source for ${field.label || field.name}`}><SelectValue placeholder="Messages field" /></SelectTrigger><SelectContent>{sourceFields.map((source) => <SelectItem key={source} value={source}>{source}</SelectItem>)}</SelectContent></Select> : null}
          {kind === "media_root" ? <div className="grid grid-cols-2 gap-2"><Select value={expression?.kind === "media_root" ? expression.source : ""} onValueChange={(source) => onChange({ kind: "media_root", source, root: expression?.kind === "media_root" ? expression.root : mediaRoot || "" })}><SelectTrigger aria-label={`Media source for ${field.label || field.name}`}><SelectValue placeholder="Filename field" /></SelectTrigger><SelectContent>{sourceFields.map((source) => <SelectItem key={source} value={source}>{source}</SelectItem>)}</SelectContent></Select><Input value={expression?.kind === "media_root" ? expression.root : mediaRoot || ""} onChange={(event) => onChange({ kind: "media_root", source: expression?.kind === "media_root" ? expression.source : sourceFields[0] || "", root: event.target.value })} placeholder="Choose source root" aria-label={`Media root for ${field.label || field.name}`} /></div> : null}
        </div>
      </div>
    </div>
  );
}

function MappingPreviewPane({ preview, semantic, scenario, loading }: { preview?: MappingPreview | null; semantic?: SemanticPreviewResponse | null; scenario?: TrainingScenarioDescriptor; loading: boolean }) {
  const [index, setIndex] = useState(0);
  const item = semantic?.items[index] ?? semantic?.items[0];
  const rawItem = preview?.items[index] ?? preview?.items[0];
  const total = semantic?.items.length ?? preview?.items.length ?? 0;
  return (
    <aside className="min-w-0 bg-bg-subtle/35">
      <div className="flex items-center justify-between border-b border-border-subtle px-4 py-3"><div><div className="text-[10px] font-medium uppercase tracking-[0.12em] text-fg-disabled">Semantic preview</div><div className="mt-0.5 text-xs font-medium text-fg">{scenario ? scenarioKindLabel(scenario) : "Training record"}</div></div>{preview ? <Badge size="sm" tone={preview.ready ? "success" : "warning"}>{preview.valid_count}/{preview.total_sampled} valid</Badge> : null}</div>
      {loading ? <LoadingState label="Updating semantic preview" /> : item ? (
        <div>
          <SemanticRecordCard item={item} />
          {(item.issues?.length ?? 0) > 0 ? <div className="border-t border-border-subtle px-4 py-3 text-[10px] text-warning">{item.issues!.map((issue) => issue.message).join(" · ")}</div> : null}
          {rawItem ? <details className="border-t border-border-subtle px-4 py-3"><summary className="cursor-pointer text-[10px] font-medium text-fg-subtle">Advanced · technical record</summary><div className="mt-3 grid gap-3"><PreviewObject label="Source fields" value={rawItem.source} /><PreviewObject label="Canonical fields" value={rawItem.canonical} /></div></details> : null}
          {total > 1 ? <div className="flex justify-between border-t border-border-subtle px-4 py-2"><Button variant="ghost" size="sm" disabled={index === 0} onClick={() => setIndex(Math.max(0, index - 1))}>Previous</Button><span className="self-center font-mono text-[10px] text-fg-subtle">{index + 1}/{total}</span><Button variant="ghost" size="sm" disabled={index + 1 >= total} onClick={() => setIndex(Math.min(total - 1, index + 1))}>Next</Button></div> : null}
        </div>
      ) : <EmptyState icon={Braces} title="Preview is waiting" detail="Connect the required fields to see the training example as the model will receive it." compact />}
    </aside>
  );
}

function SemanticRecordCard({ item }: { item: SemanticRecordPreview }) {
  const presentation = item.presentation ?? {};
  const turns = presentation.turns ?? [];
  if (item.kind === "chat" || item.kind === "tool") {
    return (
      <div>
        <SemanticHeader title={item.title} summary={item.summary} icon={item.kind === "tool" ? Wrench : MessageSquareText} />
        <div className="space-y-2 px-4 py-4">
          {turns.map((turn, index) => <div key={`${turn.role}-${index}`} className={cn("max-w-[92%] border-l-2 px-3 py-2", turn.role === "assistant" ? "ml-auto border-accent bg-accent-bg/25" : turn.role === "tool" ? "border-warning bg-warning-bg/25" : "border-border-strong bg-bg")}><div className="text-[9px] font-medium uppercase tracking-[0.12em] text-fg-disabled">{friendlyRole(turn.role)}</div><div className="mt-1 whitespace-pre-wrap text-[11px] leading-5 text-fg-muted">{turn.content || "No text content"}</div>{turn.tool_calls?.length ? <div className="mt-2 text-[10px] text-accent">{turn.tool_calls.length} tool {turn.tool_calls.length === 1 ? "call" : "calls"}</div> : null}</div>)}
          {item.kind === "tool" && !turns.length ? <SemanticValueList values={{ "Expected call": presentation.expected_calls, "Expected result": presentation.expected_results }} /> : null}
        </div>
      </div>
    );
  }
  if (item.kind === "preference") {
    return (
      <div>
        <SemanticHeader title={item.title} summary={item.summary} icon={ShieldCheck} />
        <div className="px-4 py-4">
          {presentation.prompt ? <SemanticText label="Prompt" value={presentation.prompt} /> : null}
          <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2"><SemanticChoice label="Preferred" value={presentation.chosen} positive /><SemanticChoice label="Not preferred" value={presentation.rejected} /></div>
        </div>
      </div>
    );
  }
  if (item.kind === "vlm") {
    return <div><SemanticHeader title={item.title} summary={item.summary} icon={Image} /><div className="px-4 py-4"><MediaReference icon={Image} label="Image" value={presentation.image} /><div className="mt-3 space-y-3"><SemanticText label="Prompt" value={presentation.prompt} /><SemanticText label="Expected response" value={presentation.response ?? presentation.ground_truth} /></div></div></div>;
  }
  if (item.kind === "audio") {
    return <div><SemanticHeader title={item.title} summary={item.summary} icon={AudioLines} /><div className="px-4 py-4"><MediaReference icon={AudioLines} label="Audio" value={presentation.audio} /><div className="mt-3 space-y-3"><SemanticText label={presentation.transcript ? "Transcript" : "Label"} value={presentation.transcript ?? presentation.label} />{presentation.task ? <SemanticText label="Task" value={humanize(String(presentation.task))} /> : null}</div></div></div>;
  }
  if (item.kind === "corpus") {
    return <div><SemanticHeader title={item.title} summary={item.summary} icon={BookOpen} /><div className="px-4 py-4"><div className="max-h-72 overflow-auto whitespace-pre-wrap text-[11px] leading-5 text-fg-muted">{asDisplayText(presentation.text) || "No extracted text"}</div>{presentation.source_ref ? <div className="mt-3 border-t border-border-subtle pt-3 text-[10px] text-fg-subtle">Source: {displayReference(presentation.source_ref)}</div> : null}</div></div>;
  }
  return (
    <div>
      <SemanticHeader title={item.title} summary={item.summary} icon={FileText} />
      <div className="space-y-3 px-4 py-4">
        {presentation.system ? <SemanticText label="System guidance" value={presentation.system} /> : null}
        <SemanticText label="Prompt" value={presentation.prompt} />
        <SemanticText label="Expected response" value={presentation.response ?? presentation.reference_answer ?? presentation.text} />
      </div>
    </div>
  );
}

function SemanticHeader({ title, summary, icon: Icon }: { title: string; summary: string; icon: LucideIcon }) {
  return <div className="flex gap-3 border-b border-border-subtle bg-bg px-4 py-3"><span className="grid h-8 w-8 shrink-0 place-items-center rounded-md border border-border text-fg-subtle"><Icon className="h-4 w-4" /></span><div className="min-w-0"><div className="truncate text-[12px] font-medium text-fg">{title}</div><div className="mt-0.5 text-[10px] leading-4 text-fg-subtle">{summary}</div></div></div>;
}

function SemanticText({ label, value }: { label: string; value: unknown }) {
  const text = asDisplayText(value);
  if (!text) return null;
  return <div><div className="text-[9px] font-medium uppercase tracking-[0.12em] text-fg-disabled">{label}</div><div className="mt-1 whitespace-pre-wrap text-[11px] leading-5 text-fg-muted">{text}</div></div>;
}

function SemanticChoice({ label, value, positive = false }: { label: string; value: unknown; positive?: boolean }) {
  return <div className={cn("border-l-2 px-3 py-3", positive ? "border-success bg-success-bg/35" : "border-border-strong bg-bg")}><div className={cn("text-[9px] font-medium uppercase tracking-[0.12em]", positive ? "text-success" : "text-fg-disabled")}>{label}</div><div className="mt-1 whitespace-pre-wrap text-[11px] leading-5 text-fg-muted">{asDisplayText(value) || "No response"}</div></div>;
}

function MediaReference({ icon: Icon, label, value }: { icon: LucideIcon; label: string; value: unknown }) {
  return <div className="flex min-h-20 items-center gap-3 border-y border-border-subtle bg-bg-subtle/40 px-3 py-3"><span className="grid h-9 w-9 shrink-0 place-items-center rounded-md border border-border text-fg-subtle"><Icon className="h-4 w-4" /></span><div className="min-w-0"><div className="text-[9px] font-medium uppercase tracking-[0.12em] text-fg-disabled">{label}</div><div className="mt-1 truncate text-[11px] text-fg-muted">{displayReference(value)}</div></div></div>;
}

function SemanticValueList({ values }: { values: Record<string, unknown> }) {
  return <div className="divide-y divide-border-subtle border-y border-border-subtle">{Object.entries(values).map(([label, value]) => <div key={label} className="grid grid-cols-[110px_minmax(0,1fr)] gap-3 py-2 text-[10px]"><span className="text-fg-subtle">{label}</span><span className="text-fg-muted">{asDisplayText(value) || "None"}</span></div>)}</div>;
}

function PrepareStep(context: StudioContext) {
  const { draft, setDraft, selectedScenario, preparationPlan, inspectionReadiness, inspectionReadinessLoading, recipe, recipeError } = context;
  if (!selectedScenario || !draft.mappingPlan) return <StepSurface number="05" title="Prepare the version" detail="Confirm field mapping first."><EmptyState icon={WandSparkles} title="No confirmed mapping" detail="Return to Map and confirm the canonical records." /></StepSurface>;
  const corpus = isCorpusScenario(selectedScenario);
  function updateSetting<K extends keyof PreparationSettings>(key: K, value: PreparationSettings[K]) {
    setDraft((current) => ({ ...current, preparation: { ...current.preparation, [key]: value } }));
  }
  const ratioTotal = draft.preparation.trainRatio + draft.preparation.validationRatio + draft.preparation.testRatio;
  return (
    <StepSurface
      number="05"
      title={corpus ? "Prepare the extracted corpus" : "Review how the data will be prepared"}
      detail={corpus
        ? "Keep document lineage intact, remove extraction noise, review duplicates, and create a held-out validation split before token packing."
        : "Normal mode exposes safe, ordered operations. Advanced mode preserves the exact recipe without rewriting it."}
    >
      <div className="divide-y divide-border-subtle">
        {corpus ? (
          <>
            <PreparationToggle index="01" title="Preserve headings and code blocks" detail="Keep meaningful Markdown structure while normalizing document text." checked={draft.preparation.preserveHeadings} onChange={(value) => updateSetting("preserveHeadings", value)} />
            <PreparationToggle index="02" title="Remove repeated navigation and boilerplate" detail="Strip common headers, footers, and navigation that would otherwise dominate the corpus." checked={draft.preparation.stripBoilerplate} onChange={(value) => updateSetting("stripBoilerplate", value)} />
            <PreparationToggle index="03" title="Quarantine extraction failures" detail="Keep encrypted, image-only, empty, or unsupported documents out of training and available for review." checked={draft.preparation.quarantineExtractionFailures} onChange={(value) => updateSetting("quarantineExtractionFailures", value)} />
            <PreparationToggle index="04" title="Preserve document boundaries" detail="Group paragraphs from the same document so one source cannot cross the training and validation boundary." checked={draft.preparation.preserveDocumentBoundaries} onChange={(value) => updateSetting("preserveDocumentBoundaries", value)} />
            <PreparationToggle index="05" title="Remove exact and near duplicates" detail="Use content hashes plus the reviewed fuzzy threshold before splitting." checked={draft.preparation.exactDedup && draft.preparation.fuzzyDedup} onChange={(value) => { updateSetting("exactDedup", value); updateSetting("fuzzyDedup", value); }} />
          </>
        ) : (
          <>
            <PreparationToggle index="01" title="Normalize whitespace" detail="Trim text and collapse repeated whitespace while preserving record meaning." checked={draft.preparation.normalizeWhitespace} onChange={(value) => updateSetting("normalizeWhitespace", value)} />
            <PreparationToggle index="02" title="Validate and quarantine" detail="Keep malformed records out of training and retain them for inspection." checked={draft.preparation.validate} onChange={(value) => updateSetting("validate", value)} />
            <PreparationToggle index="03" title="Remove exact duplicates" detail="Deduplicate canonical content deterministically before splitting." checked={draft.preparation.exactDedup} onChange={(value) => updateSetting("exactDedup", value)} />
          </>
        )}
        <div className="grid gap-4 px-5 py-4 sm:grid-cols-[minmax(0,1fr)_280px]"><div><div className="flex items-center gap-2 text-[12px] font-medium text-fg"><span className="font-mono text-[9px] text-fg-disabled">04</span>Split the records</div><div className="mt-1 text-[11px] leading-5 text-fg-muted">Seed 42 keeps unchanged inputs in the same order and split. Validation is preserved exactly for training.</div></div><div className="grid grid-cols-3 gap-2">{(["trainRatio", "validationRatio", "testRatio"] as const).map((key) => <Field key={key} label={key === "trainRatio" ? "Train %" : key === "validationRatio" ? "Validation %" : "Test %"}><Input type="number" min={0} max={100} value={draft.preparation[key]} onChange={(event) => updateSetting(key, Number(event.target.value))} mono /></Field>)}</div>{ratioTotal !== 100 ? <div className="sm:col-start-2 text-[10px] text-danger">Splits must total 100% (currently {ratioTotal}%).</div> : null}</div>
        <PreparationToggle index="05" title="Check contamination" detail="Report overlap across train, validation, test, and protected evidence splits." checked={draft.preparation.contamination} onChange={(value) => updateSetting("contamination", value)} />
        {["image", "audio"].includes(selectedScenario.modality) ? <PreparationToggle index="06" title="Keep shared media together" detail="Group identical image or audio hashes so one asset cannot cross into a held-out split." checked={draft.preparation.groupMedia} onChange={(value) => updateSetting("groupMedia", value)} /> : null}
      </div>
      <ReadinessReportPanel
        report={inspectionReadiness}
        loading={inspectionReadinessLoading}
        title="Dataset readiness"
        onAction={(target) => setDraft((current) => ({ ...current, step: remediationTargetStep(target) ?? current.step }))}
      />
      <div className="border-t border-border-subtle">
        <button type="button" onClick={() => setDraft((current) => ({ ...current, advancedRecipe: !current.advancedRecipe, rawRecipe: !current.advancedRecipe && !current.rawRecipe ? JSON.stringify(recipe, null, 2) : current.rawRecipe }))} className="flex w-full items-center justify-between px-5 py-3 text-left"><span><span className="block text-[11px] font-medium text-fg">Advanced recipe</span><span className="mt-0.5 block text-[10px] text-fg-subtle">Inspect or edit YAML or JSON. Halo Forge preserves the entered text while this draft is open.</span></span><ChevronDown className={cn("h-4 w-4 text-fg-subtle transition-transform", draft.advancedRecipe && "rotate-180")} /></button>
        {draft.advancedRecipe ? <div className="border-t border-border-subtle px-5 py-4"><textarea value={draft.rawRecipe} onChange={(event) => setDraft((current) => ({ ...current, rawRecipe: event.target.value }))} rows={18} spellCheck={false} className="w-full resize-y rounded-md border border-border bg-bg-subtle px-3 py-3 font-mono text-[11px] leading-5 text-fg focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/25" aria-label="Advanced dataset recipe YAML or JSON" />{recipeError ? <div className="mt-2 text-[10px] text-danger">{recipeError}</div> : null}</div> : null}
      </div>
      <div className="grid gap-px border-t border-border-subtle bg-border-subtle sm:grid-cols-4"><Readout label="Estimated accepted" value={formatInteger(preparationPlan?.estimates?.accepted)} sampled /><Readout label="Quarantined" value={formatInteger(preparationPlan?.estimates?.quarantined)} sampled /><Readout label="Duplicates" value={formatInteger(preparationPlan?.estimates?.duplicates)} sampled /><Readout label="Ordered steps" value={String(recipe.steps.length)} /></div>
    </StepSurface>
  );
}

function VersionStep(context: StudioContext) {
  const { draft, setDraft, selectedScenario, inspection, recipe, preparationPlan, inspectionReadiness, buildMutation, buildJob, buildComplete, builtVersionId } = context;
  const progress = buildJob?.progress_percent;
  return (
    <StepSurface number="06" title="Publish an immutable dataset version" detail="Review the resolved plan. Publication is explicit and never changes your original source.">
      <div className="grid gap-px bg-border-subtle lg:grid-cols-[minmax(0,1fr)_300px]">
        <div className="space-y-4 bg-bg px-5 py-5">
          <div className="grid gap-3 sm:grid-cols-2"><Field label="Dataset name"><Input value={draft.datasetName} onChange={(event) => setDraft((current) => ({ ...current, datasetName: event.target.value }))} placeholder={selectedScenario ? defaultDatasetName(selectedScenario, draft) : "Training dataset"} /></Field><Field label="Description"><Input value={draft.datasetDescription} onChange={(event) => setDraft((current) => ({ ...current, datasetDescription: event.target.value }))} placeholder="Purpose or collection notes" /></Field></div>
          <div className="border-y border-border-subtle"><SummaryRow label="Scenario" value={selectedScenario?.label || "—"} /><SummaryRow label="Source records" value={formatInteger(inspection?.row_count)} /><SummaryRow label="Mapping" value={`${Object.keys(draft.mappingPlan?.mappings ?? {}).length} canonical fields confirmed`} /><SummaryRow label="Preparation" value={`${recipe.steps.length} ordered steps · seed 42`} /><SummaryRow label="Assets" value={["image", "audio"].includes(selectedScenario?.modality || "") ? "Referenced and hash-grouped" : "No binary assets"} /></div>
          {preparationPlan?.warnings?.length ? <ErrorBanner title="Preparation warnings" detail={preparationPlan.warnings.join(" ")} tone="warning" /> : null}
          {inspectionReadiness?.ready === false ? <ErrorBanner title="Dataset readiness needs attention" detail={inspectionReadiness.blockers[0]?.message || "Return to Prepare and complete the recommended remediation before publishing."} /> : null}
          {buildMutation.isError ? <ErrorBanner title="Version could not be published" detail={(buildMutation.error as Error).message} /> : null}
          {buildJob && !buildComplete ? <ProgressStrip label={buildJob.stage || "Building immutable version"} progress={progress ?? undefined} detail={buildJob.error || undefined} /> : null}
          {buildComplete ? <div className="border-l-2 border-success bg-success-bg px-4 py-3"><div className="flex items-center gap-2 text-xs font-medium text-success"><CheckCircle2 className="h-4 w-4" />Dataset version is ready</div><p className="mt-1 text-[11px] leading-5 text-fg-muted">Exact counts, split files, checksums, and transformation provenance were published atomically.</p><div className="mt-3 flex flex-wrap gap-2"><Button variant="secondary" size="sm" asChild><Link to="/datasets/$datasetId/versions/$versionId" params={{ datasetId: draft.datasetId, versionId: builtVersionId }} search={{ split: "train" }}>Open version</Link></Button><Button variant="primary" size="sm" onClick={() => setDraft((current) => ({ ...current, step: "train", versionId: builtVersionId }))}>Continue to training<ArrowRight /></Button></div></div> : null}
        </div>
        <aside className="bg-bg-subtle/35 px-5 py-5"><div className="text-[10px] font-medium uppercase tracking-[0.12em] text-fg-disabled">Publication contract</div><ul className="mt-3 space-y-3">{["Source remains unchanged", "Recipe and source fingerprints are stored", "Failed builds are never exposed as versions", "Rebuilding changed data creates a new identity"].map((item) => <li key={item} className="flex gap-2 text-[11px] leading-5 text-fg-muted"><Check className="mt-0.5 h-3.5 w-3.5 shrink-0 text-success" />{item}</li>)}</ul></aside>
      </div>
      {!buildComplete ? <div className="flex justify-end border-t border-border-subtle bg-bg-subtle/45 px-5 py-4"><Button variant="primary" size="lg" disabled={buildMutation.isPending || Boolean(buildJob && !["failed", "cancelled"].includes(buildJob.status)) || !draft.mappingConfirmed || inspectionReadiness?.ready === false} onClick={() => buildMutation.mutate()}>{buildMutation.isPending || (buildJob && !["failed", "cancelled", "completed", "succeeded"].includes(buildJob.status)) ? <Loader2 className="animate-spin" /> : <ShieldCheck />}Publish version</Button></div> : null}
    </StepSurface>
  );
}

function TrainStep(context: StudioContext) {
  // Every normal guided scenario, including continued pretraining, uses the
  // same backend-owned recommendation and measured-capacity workflow. The
  // older CPT controls remain available through Advanced launch surfaces.
  if (isCorpusScenario(context.selectedScenario) && context.draft.advancedRecipe) {
    return <CorpusTrainStep {...context} />;
  }
  return <ProofTrainStep {...context} />;
}

function CorpusTrainStep(context: StudioContext) {
  const { draft, setDraft, selectedScenario, builtVersionId, inspection, preparationPlan, backendName } = context;
  const models = useTrainingModels({ mode: "cpt", modality: "text" });
  const telemetry = useTelemetry();
  const workstation = useBackendInfo();
  const workspace = useWorkspaceInfo();
  const runtimeModels = useMemo(() => {
    const backendId = workstation.data?.name;
    const totalMemory = telemetry.data?.sys_mem_total_gb;
    return [...(models.data?.items ?? [])]
      .filter((model) => !backendId || !model.backend_support?.length || model.backend_support.includes(backendId))
      .filter((model) => !totalMemory || !model.estimated_memory_gb || model.estimated_memory_gb <= totalMemory * 0.9)
      .sort((left, right) => Number(right.recommended_first_run) - Number(left.recommended_first_run));
  }, [models.data?.items, telemetry.data?.sys_mem_total_gb, workstation.data?.name]);
  const selectedModel = draft.model || runtimeModels.find((model) => model.recommended_first_run)?.id || selectedScenario?.recommended_model || "";
  const profileQuery = useQuery({
    queryKey: ["dataset-versions", builtVersionId, "corpus-profile"],
    queryFn: () => api.corpusProfile(builtVersionId),
    enabled: Boolean(builtVersionId),
    retry: false,
  });
  const profile = profileQuery.data ?? localCorpusProfile(inspection, preparationPlan);
  const packingRequest = useMemo<CorpusPackingRequest | null>(() => {
    if (!selectedModel || !draft.cptAdaptation) return null;
    return {
      model: selectedModel,
      adaptation: draft.cptAdaptation,
      max_sequence_length: draft.cptMaxSequenceLength,
      packing: draft.cptPacking,
      budget_mode: draft.cptBudgetMode,
      target_tokens: draft.cptBudgetMode === "tokens" ? draft.cptTargetTokens : null,
      corpus_passes: draft.cptBudgetMode === "passes" ? draft.cptCorpusPasses : null,
      effective_batch_size: 1,
      seed: 42,
    };
  }, [draft.cptAdaptation, draft.cptBudgetMode, draft.cptCorpusPasses, draft.cptMaxSequenceLength, draft.cptPacking, draft.cptTargetTokens, selectedModel]);
  const packingQuery = useQuery({
    queryKey: ["dataset-versions", builtVersionId, "packing-plan", packingRequest],
    queryFn: () => api.corpusPackingPlan(builtVersionId, packingRequest!),
    enabled: Boolean(builtVersionId && packingRequest),
    retry: false,
    refetchInterval: (query) => packingPreparationState(query.state.data) ? 1_000 : false,
  });
  const exactPackingPlan = unwrapCorpusPackingPlan(packingQuery.data);
  const packingPlan = exactPackingPlan ?? (packingRequest ? localCorpusPackingPlan(profile, packingRequest) : null);
  const packingPreparation = packingPreparationState(packingQuery.data);
  const trainingConfig = useMemo<CorpusTrainingConfig | null>(() => {
    if (!packingRequest || !builtVersionId) return null;
    return {
      dataset_version_id: builtVersionId,
      ...packingRequest,
      output: workspace.data?.default_run_root || null,
    };
  }, [builtVersionId, packingRequest, workspace.data?.default_run_root]);
  const preflight = useQuery({
    queryKey: ["dataset-versions", builtVersionId, "cpt-preflight", trainingConfig],
    queryFn: async () => {
      try {
        return await api.cptPreflight(trainingConfig!);
      } catch (error) {
        if (!(error instanceof ApiError) || ![404, 405].includes(error.status)) throw error;
        return api.trainingPreflight({
          mode: "cpt",
          ...trainingConfig!,
          model_name: trainingConfig!.model,
          max_seq_length: trainingConfig!.max_sequence_length,
        });
      }
    },
    enabled: Boolean(trainingConfig && !packingPreparation),
    retry: false,
    refetchInterval: (query) => ["preparing", "preparing_dataset", "queued", "running"].includes(String(query.state.data?.status || "")) && query.state.data?.ready !== true ? 1_000 : false,
  });
  const launch = useMutation({
    mutationFn: async () => {
      try {
        return await api.launchCpt(trainingConfig!);
      } catch (error) {
        if (!(error instanceof ApiError) || ![404, 405].includes(error.status)) throw error;
        return api.trainingLaunch({
          mode: "cpt",
          ...trainingConfig!,
          model_name: trainingConfig!.model,
          max_seq_length: trainingConfig!.max_sequence_length,
        });
      }
    },
    onSuccess: (result) => {
      if (result.status === "preparing_dataset" || result.status === "preparing" || result.ready === false) return;
      setDraft((current) => ({ ...current, model: selectedModel, fullRunId: String(result.run_id || result.id || "") }));
    },
  });
  const readiness = preflight.data && "readiness" in preflight.data
    ? (preflight.data as { readiness?: DatasetReadiness }).readiness
    : undefined;
  const preflightErrors = preflight.data?.errors ?? [];
  const preflightPreparing = ["preparing", "preparing_dataset", "queued", "running"].includes(String(preflight.data?.status || "")) && preflight.data?.ready !== true;
  const launchPreparing = Boolean(launch.data && (launch.data.status === "preparing_dataset" || launch.data.status === "preparing" || launch.data.ready === false));
  const preflightReady = preflight.data?.ok === true && preflight.data.ready !== false && readiness?.ready !== false && preflightErrors.length === 0;
  const controlsReady = Boolean(selectedModel && draft.cptAdaptation && trainingConfig);
  const packingLabel = draft.cptPacking === "paragraph_eos_non_overlap_v1"
    ? "Pack paragraphs with EOS boundaries"
    : humanize(draft.cptPacking);
  return (
    <StepSurface number="07" title="Plan continued pretraining" detail="Choose the model, adaptation method, sequence length, and a bounded token or corpus-pass budget. Packing is previewed before anything launches.">
      {!builtVersionId ? <EmptyState icon={Database} title="Publish a corpus version first" detail="Continued pretraining always binds an immutable document corpus with retained provenance." /> : (
        <div className="grid gap-px bg-border-subtle lg:grid-cols-[minmax(0,1fr)_320px]">
          <div className="space-y-5 bg-bg px-5 py-5">
            <div className="grid gap-3 sm:grid-cols-2">
              <Field label="Model" hint="Choose a causal language model supported by the active training backend.">
                <Input list="cpt-model-options" value={selectedModel} onChange={(event) => setDraft((current) => ({ ...current, model: event.target.value }))} placeholder="Model name or repository" aria-label="Continued pretraining model" />
                <datalist id="cpt-model-options">{runtimeModels.map((model) => <option key={model.id} value={model.id}>{model.label}</option>)}</datalist>
              </Field>
              <Field label="Adaptation method" hint="Required. LoRA updates adapters; full updates every model weight.">
                <Select value={draft.cptAdaptation || undefined} onValueChange={(value) => setDraft((current) => ({ ...current, cptAdaptation: value as "lora" | "full" }))}>
                  <SelectTrigger aria-label="Required adaptation method"><SelectValue placeholder="Choose LoRA or full" /></SelectTrigger>
                  <SelectContent><SelectItem value="lora">LoRA adapters</SelectItem><SelectItem value="full">Full weight update</SelectItem></SelectContent>
                </Select>
              </Field>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <Field label="Sequence length" hint="Fixed for tokenization and packing; longer sequences use more memory.">
                <Select value={String(draft.cptMaxSequenceLength)} onValueChange={(value) => setDraft((current) => ({ ...current, cptMaxSequenceLength: Number(value) }))}>
                  <SelectTrigger aria-label="Maximum sequence length"><SelectValue /></SelectTrigger>
                  <SelectContent>{[512, 1024, 2048, 4096, 8192].map((value) => <SelectItem key={value} value={String(value)}>{formatInteger(value)} tokens</SelectItem>)}</SelectContent>
                </Select>
              </Field>
              <Field label="Packing" hint="Deterministic, non-overlapping blocks with document provenance retained.">
                <Select value={draft.cptPacking} onValueChange={(value) => setDraft((current) => ({ ...current, cptPacking: value }))}>
                  <SelectTrigger aria-label="Corpus packing strategy"><SelectValue /></SelectTrigger>
                  <SelectContent><SelectItem value="paragraph_eos_non_overlap_v1">Pack paragraphs with EOS boundaries</SelectItem></SelectContent>
                </Select>
              </Field>
            </div>
            <div className="border-y border-border-subtle">
              <div className="flex flex-wrap items-center justify-between gap-3 px-3 py-3"><div><div className="text-[11px] font-medium text-fg">Training budget</div><div className="mt-0.5 text-[10px] text-fg-subtle">Set one explicit limit. Halo Forge does not infer an open-ended budget.</div></div><div className="flex rounded-md border border-border bg-bg p-0.5"><button type="button" aria-pressed={draft.cptBudgetMode === "tokens"} onClick={() => setDraft((current) => ({ ...current, cptBudgetMode: "tokens" }))} className={cn("rounded px-2.5 py-1 text-[10px]", draft.cptBudgetMode === "tokens" ? "bg-accent-bg text-accent" : "text-fg-muted")}>Token budget</button><button type="button" aria-pressed={draft.cptBudgetMode === "passes"} onClick={() => setDraft((current) => ({ ...current, cptBudgetMode: "passes" }))} className={cn("rounded px-2.5 py-1 text-[10px]", draft.cptBudgetMode === "passes" ? "bg-accent-bg text-accent" : "text-fg-muted")}>Corpus passes</button></div></div>
              <div className="px-3 pb-4">{draft.cptBudgetMode === "tokens" ? <Field label="Target training tokens" hint="The packer stops at or below this reviewed ceiling."><Input type="number" min={1} step={1000} value={draft.cptTargetTokens} onChange={(event) => setDraft((current) => ({ ...current, cptTargetTokens: Math.max(1, Number(event.target.value)) }))} mono /></Field> : <Field label="Corpus passes" hint="One pass sees each accepted training token once before packing limits."><Input type="number" min={0.01} step={0.25} value={draft.cptCorpusPasses} onChange={(event) => setDraft((current) => ({ ...current, cptCorpusPasses: Math.max(0.01, Number(event.target.value)) }))} mono /></Field>}</div>
            </div>
            {draft.cptAdaptation === "full" ? <ErrorBanner title="Full weight update selected" detail="This updates every model weight and typically needs substantially more memory and checkpoint storage than LoRA. Preflight must verify the choice." tone="warning" /> : null}
            <div className="grid gap-px border-y border-border-subtle bg-border-subtle sm:grid-cols-4">
              <Readout label="Documents" value={formatInteger(profile.document_count)} />
              <Readout label="Paragraphs" value={formatInteger(profile.paragraph_count)} />
              <Readout label="Training tokens" value={formatInteger(packingPlan?.train_tokens)} sampled={!exactPackingPlan} />
              <Readout label="Packed sequences" value={formatInteger(packingPlan?.train_blocks)} sampled={!exactPackingPlan} />
            </div>
            {packingPreparation ? <ProgressStrip label="Preparing the tokenizer-aware packing plan" progress={packingPreparation.progress ?? undefined} detail={packingPreparation.message || "This is a durable preparation job. Track it in Activity while local estimates remain clearly marked."} /> : null}
            {packingPlan ? <div className="border-l-2 border-accent bg-accent-bg/25 px-4 py-3"><div className="text-xs font-medium text-fg">{exactPackingPlan ? "Packing preview" : "Estimated packing preview"}</div><div className="mt-1 text-[11px] leading-5 text-fg-muted">{packingLabel} into {formatInteger(packingPlan.max_sequence_length)}-token sequences · {percent(packingPlan.utilization)} utilization · about {formatInteger(packingPlan.estimated_steps)} optimizer steps.</div>{!exactPackingPlan ? <div className="mt-2 text-[10px] text-fg-subtle">{packingQuery.isError ? "Showing a local estimate until the tokenizer-aware packing route is available." : "Exact tokenizer statistics will replace this estimate when the durable packing job completes."}</div> : null}</div> : null}
            {preflight.isLoading ? <ProgressStrip label="Checking corpus, model, and workstation readiness" /> : preflight.isError ? <ErrorBanner title="CPT preflight is unavailable" detail={(preflight.error as Error).message} tone="warning" /> : preflightPreparing ? <ProgressStrip label="Preparing continued-pretraining artifacts" detail={preflight.data?.message || "Track the durable preparation job in Activity."} /> : readiness ? <ReadinessReportPanel report={readiness} title="Continued pretraining readiness" /> : preflightErrors.length ? <div className="space-y-2">{preflightErrors.map((message, index) => <ErrorBanner key={`${message}-${index}`} title={message} detail={preflight.data?.suggested_fixes?.[index] || "Resolve this item, then run preflight again."} />)}</div> : preflightReady ? <div className="border-l-2 border-success bg-success-bg px-4 py-3"><div className="text-xs font-medium text-success">Ready to train the corpus</div><div className="mt-1 text-[11px] leading-5 text-fg-muted">The immutable corpus, tokenizer-aware packing plan, selected adaptation, model access, and workstation resources are compatible.</div></div> : controlsReady ? <Hint>Preflight will confirm the selected model and budget.</Hint> : <Hint>Choose a model and explicitly select LoRA or full adaptation to continue.</Hint>}
            {launch.isError ? <ErrorBanner title="Continued pretraining did not start" detail={(launch.error as Error).message} /> : null}
            {launchPreparing ? <ProgressStrip label="Preparing continued pretraining" detail={launch.data?.message || "Track the durable launch preparation in Activity."} /> : null}
            {draft.fullRunId ? <div className="border-l-2 border-success bg-success-bg px-4 py-3"><div className="text-xs font-medium text-success">Continued pretraining started</div><div className="mt-2"><Button size="sm" variant="secondary" asChild><Link to="/runs/$runId" params={{ runId: draft.fullRunId }}>Open run</Link></Button></div></div> : null}
          </div>
          <aside className="bg-bg-subtle/35 px-5 py-5"><div className="text-[10px] font-medium uppercase tracking-[0.12em] text-fg-disabled">Confirmation</div><div className="mt-3 text-sm font-medium text-fg">{selectedScenario?.label || "Document corpus"}</div><div className="mt-1 break-words text-[10px] leading-5 text-fg-subtle">{selectedModel || "Choose a model"} · {draft.cptAdaptation ? draft.cptAdaptation === "lora" ? "LoRA adapters" : "Full weight update" : "adaptation not selected"}</div><div className="mt-5"><Button variant="primary" size="lg" className="w-full" disabled={!preflightReady || launch.isPending || Boolean(draft.fullRunId)} onClick={() => launch.mutate()}>{launch.isPending ? <Loader2 className="animate-spin" /> : <Play />}Start continued pretraining</Button></div><p className="mt-3 text-[10px] leading-4 text-fg-subtle">Nothing launches automatically. The run records the corpus version, model, explicit adaptation, packing plan, sequence length, and budget.</p><div className="mt-5 border-t border-border-subtle pt-4 text-[10px] text-fg-subtle">Backend: {backendName || "automatic"} · seed 42</div></aside>
        </div>
      )}
    </StepSurface>
  );
}

function ProofTrainStep(context: StudioContext) {
  const { draft, setDraft, selectedScenario, builtVersionId, backendName } = context;
  const queryClient = useQueryClient();
  const trainerMode = (draft.trainerMode || (selectedScenario ? preferredTrainerMode(selectedScenario) : "sft")) as TrainingMode;
  const models = useTrainingModels({ mode: trainerMode });
  const workstation = useBackendInfo();
  const telemetry = useTelemetry();
  const workspace = useWorkspaceInfo();
  const runtimeModels = useMemo(() => {
    const backendId = workstation.data?.name;
    const totalMemory = telemetry.data?.sys_mem_total_gb;
    return [...(models.data?.items ?? [])]
      .filter((model) => !backendId || !model.backend_support?.length || model.backend_support.includes(backendId) || (backendId === "mlx" && model.backend_support.includes("mlx")))
      .filter((model) => !totalMemory || !model.estimated_memory_gb || model.estimated_memory_gb <= totalMemory * 0.8)
      .sort((left, right) => (left.estimated_memory_gb ?? Number.MAX_SAFE_INTEGER) - (right.estimated_memory_gb ?? Number.MAX_SAFE_INTEGER));
  }, [models.data?.items, telemetry.data?.sys_mem_total_gb, workstation.data?.name]);
  const recommended = runtimeModels.find((model) => model.recommended_first_run) ?? runtimeModels[0];
  const scenarioRecommended = runtimeModels.find((model) => model.id === selectedScenario?.recommended_model)?.id;
  const fallbackModel = draft.model || scenarioRecommended || recommended?.id || "";
  const verifierRequired = ["raft", "grpo"].includes(String(trainerMode));
  const verifierProfiles = useQuery({
    queryKey: ["verifier-profiles", "guided-own-data", trainerMode],
    queryFn: () => api.listVerifierProfiles({ modality: "text", qualification: "pass", limit: 100 }),
    enabled: verifierRequired,
    staleTime: 30_000,
  });
  const qualifiedVerifiers = useMemo(() => (verifierProfiles.data?.items ?? [])
    .map((profile) => ({ profile, revision: profile.latest_revision }))
    .filter((item) => Boolean(item.revision?.id && item.revision.runtime_compatible !== false && ["candidate", "approved"].includes(String(item.revision.alias)))), [verifierProfiles.data?.items]);
  const selectedVerifierRevisionId = verifierRequired
    ? draft.verifierRevisionId || qualifiedVerifiers[0]?.revision?.id || ""
    : "";
  const planRecommendation = useQuery({
    queryKey: ["training-plan", "recommend", builtVersionId, trainerMode, selectedScenario?.revision_id, selectedVerifierRevisionId],
    queryFn: () => api.recommendTrainingPlan({
      dataset_version_id: builtVersionId,
      scenario_revision_id: selectedScenario?.revision_id,
      trainer_mode: trainerMode,
      ...(selectedVerifierRevisionId ? { verifier_profile_revision_id: selectedVerifierRevisionId } : {}),
    }),
    enabled: Boolean(builtVersionId && (!verifierRequired || selectedVerifierRevisionId)),
    retry: false,
    staleTime: Infinity,
  });
  const recommendedPlanRevision = planRecommendation.data?.revision;
  const selectedPlanRevision = useQuery({
    queryKey: ["training-plan-revisions", draft.trainingPlanRevisionId],
    queryFn: () => api.trainingPlanRevision(draft.trainingPlanRevisionId),
    enabled: Boolean(
      draft.trainingPlanRevisionId
      && draft.trainingPlanRevisionId !== recommendedPlanRevision?.id,
    ),
    retry: false,
  });
  const planRevision = selectedPlanRevision.data ?? recommendedPlanRevision;
  const selectedModel = draft.model || planRevision?.model_id || fallbackModel;
  const modelAccess = (planRevision?.definition.model_access ?? {}) as Record<string, unknown>;
  const modelAccessNote = [modelAccess.download_note, modelAccess.license_note]
    .filter((value): value is string => typeof value === "string" && value.trim().length > 0)
    .join(" ");
  useEffect(() => {
    if (!planRecommendation.data) return;
    setDraft((current) => ({
      ...current,
      model: current.model || planRecommendation.data!.revision.model_id,
      trainingPlanId: planRecommendation.data!.plan.id,
      trainingPlanRevisionId: current.trainingPlanRevisionId || planRecommendation.data!.revision.id,
    }));
  }, [planRecommendation.data, setDraft]);
  const chooseAlternative = useMutation({
    mutationFn: (modelId: string) => api.chooseTrainingPlanAlternative(
      planRevision!.id,
      modelId,
      "Operator selected another compatible model after reviewing the recommendation.",
    ),
    onSuccess: (recommendation) => setDraft((current) => ({
      ...current,
      model: recommendation.revision.model_id,
      trainingPlanId: recommendation.plan.id,
      trainingPlanRevisionId: recommendation.revision.id,
      modelPreparationId: "",
      capacityCheckId: "",
    })),
  });
  const preparePlan = useMutation({
    mutationFn: async () => {
      const revisionId = draft.trainingPlanRevisionId || planRevision?.id;
      if (!revisionId) throw new Error("The recommended plan is not ready yet.");
      await api.confirmTrainingPlan(revisionId, { download_confirmed: true });
      return api.prepareTrainingPlanModel(revisionId);
    },
    onSuccess: (preparation) => setDraft((current) => ({ ...current, modelPreparationId: preparation.id })),
  });
  const modelPreparation = useQuery({
    queryKey: ["model-preparations", draft.modelPreparationId],
    queryFn: () => api.modelPreparation(draft.modelPreparationId),
    enabled: Boolean(draft.modelPreparationId),
    retry: false,
    refetchInterval: (query) => ["queued", "running"].includes(String(query.state.data?.status || "")) ? 1_000 : false,
  });
  const capacity = useMutation({
    mutationFn: (revisionId: string) => api.createTrainingCapacityCheck(revisionId),
    onSuccess: (check) => setDraft((current) => ({ ...current, capacityCheckId: check.id })),
  });
  useEffect(() => {
    const preparation = modelPreparation.data;
    if (!preparation || preparation.status !== "completed" || draft.capacityCheckId || capacity.isPending) return;
    const resolvedRevision = preparation.plan_revision_id;
    setDraft((current) => ({ ...current, trainingPlanRevisionId: resolvedRevision }));
    capacity.mutate(resolvedRevision);
  }, [capacity, draft.capacityCheckId, modelPreparation.data, setDraft]);
  const capacityCheck = useQuery({
    queryKey: ["training-capacity-checks", draft.capacityCheckId],
    queryFn: () => api.trainingCapacityCheck(draft.capacityCheckId),
    enabled: Boolean(draft.capacityCheckId),
    retry: false,
    refetchInterval: (query) => ["queued", "running"].includes(String(query.state.data?.status || "")) ? 1_000 : false,
  });
  const planReadinessQuery = useQuery({
    queryKey: ["training-plan-revisions", draft.trainingPlanRevisionId, "readiness"],
    queryFn: () => api.trainingPlanReadiness(draft.trainingPlanRevisionId),
    enabled: Boolean(
      draft.trainingPlanRevisionId
      && ["ready", "ready_with_adjustment"].includes(String(capacityCheck.data?.status || "")),
    ),
    retry: false,
  });
  const retryPreparation = useMutation({
    mutationFn: async () => {
      const capacityStatus = String(capacityCheck.data?.status || "");
      if (draft.capacityCheckId && ["blocked", "failed", "cancelled", "stale"].includes(capacityStatus)) {
        return api.retryTrainingCapacityCheck(
          draft.capacityCheckId,
          "Retry requested after reviewing the capacity-check remedy.",
        );
      }
      if (!draft.modelPreparationId) throw new Error("No preparation is available to retry.");
      return api.retryModelPreparation(
        draft.modelPreparationId,
        "Retry requested after reviewing the model-preparation blocker.",
      );
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["model-preparations", draft.modelPreparationId] });
      queryClient.invalidateQueries({ queryKey: ["training-capacity-checks", draft.capacityCheckId] });
    },
  });
  const readiness = useQuery<DatasetReadiness>({
    queryKey: ["dataset-versions", builtVersionId, "readiness", trainerMode, selectedModel, selectedVerifierRevisionId],
    queryFn: () => api.datasetVersionReadiness(builtVersionId, trainerMode, selectedModel, selectedVerifierRevisionId),
    enabled: Boolean(builtVersionId && selectedModel),
    retry: false,
  });
  const proofBudget = Number(
    planRevision?.definition.max_samples
      ?? selectedScenario?.proof_run?.max_samples
      ?? (["image", "audio"].includes(selectedScenario?.modality || "") ? 50 : 200),
  );
  const preflight = useQuery({
    queryKey: ["training-plan-revisions", draft.trainingPlanRevisionId, "proof-preflight"],
    queryFn: () => api.trainingPreflight({
      training_plan_revision_id: draft.trainingPlanRevisionId,
      ...(workspace.data?.default_run_root ? { output_root: workspace.data.default_run_root } : {}),
    }),
    enabled: Boolean(
      draft.trainingPlanRevisionId
      && ["ready", "ready_with_adjustment"].includes(String(capacityCheck.data?.status || ""))
      && readiness.data?.ready,
    ),
    retry: false,
    // The first managed preflight may enqueue tokenizer-aware artifact
    // rendering. Re-run the same preflight until that immutable artifact is
    // verified; otherwise the launch button would remain disabled after the
    // background job completed.
    refetchInterval: (query) => query.state.data?.status === "preparing_dataset" ? 1_000 : false,
  });
  const proof = useMutation({
    mutationFn: () => api.launchTrainingPlanProof(draft.trainingPlanRevisionId, {
      ...(workspace.data?.default_run_root ? { output_root: workspace.data.default_run_root } : {}),
    }),
    onSuccess: (result) => {
      if (result.status === "preparing_dataset" || result.ready === false) {
        // A preparation acknowledgement is a dataset job, not a training run.
        // Never poll /runs/{id} until the launch endpoint returns a real run.
        queryClient.invalidateQueries({ queryKey: ["dataset-versions", builtVersionId, "proof-preflight"] });
        return;
      }
      setDraft((current) => ({ ...current, model: selectedModel, proofRunId: String(result.run_id || result.id || "") }));
    },
  });
  const proofRun = useQuery({
    queryKey: ["runs", draft.proofRunId, "proof"],
    queryFn: () => api.runDetail(draft.proofRunId),
    enabled: Boolean(draft.proofRunId),
    retry: false,
    refetchInterval: (query) => query.state.data && ["completed", "failed", "cancelled"].includes(String(query.state.data.status)) ? false : 2_000,
  });
  const assessOutcome = useMutation({
    mutationFn: () => api.prepareTrainingOutcome(draft.proofRunId, {
      scenario_revision_id: selectedScenario?.revision_id,
    }),
    onSuccess: (preparation) => {
      if (preparation.assessment?.id) {
        setDraft((current) => ({
          ...current,
          outcomeAssessmentId: preparation.assessment!.id,
        }));
      }
    },
  });
  const outcome = useQuery({
    queryKey: ["outcome-assessments", draft.outcomeAssessmentId],
    queryFn: () => api.trainingOutcome(draft.outcomeAssessmentId),
    enabled: Boolean(draft.outcomeAssessmentId),
    retry: false,
    refetchInterval: (query) => query.state.data && ["queued", "running"].includes(query.state.data.status) ? 1_500 : false,
  });
  const outcomeGuidance = useQuery({
    queryKey: ["guidance", "training_outcome", draft.outcomeAssessmentId],
    queryFn: () => api.actionableGuidance("training_outcome", draft.outcomeAssessmentId),
    enabled: Boolean(draft.outcomeAssessmentId && outcome.data),
    retry: false,
    refetchInterval: outcome.data && ["queued", "running"].includes(outcome.data.status) ? 1_500 : false,
  });
  const outcomeSupportsFullRun = Boolean(
    outcome.data
    && outcome.data.technical_status === "verified"
    && outcome.data.status === "improved",
  );
  const full = useMutation({
    mutationFn: () => api.launchFullRunFromProof(
      draft.proofRunId,
      outcomeSupportsFullRun
        ? { assessment_id: draft.outcomeAssessmentId }
        : { override_reason: draft.outcomeOverrideReason || "" },
    ),
    onSuccess: (result) => setDraft((current) => ({ ...current, fullRunId: String(result.run_id || result.id || "") })),
  });
  const proofComplete = ["completed", "succeeded"].includes(String(proofRun.data?.status ?? (proof.data?.status === "completed" ? "completed" : "")));
  const blockers = readiness.data?.blockers ?? [];
  const preflightErrors = preflight.data?.errors ?? [];
  const readinessUnavailable = readiness.isError || (readiness.isSuccess && !readiness.data);
  const preflightUnavailable = preflight.isError || (preflight.isSuccess && !preflight.data);
  const artifactPreparing = preflight.data?.status === "preparing_dataset" || preflight.data?.ready === false;
  const planReady = ["ready", "ready_with_adjustment"].includes(String(capacityCheck.data?.status || ""));
  const launchReady = planReady
    && readiness.data?.ready === true
    && blockers.length === 0
    && preflight.data?.ready === true
    && !artifactPreparing;
  const preparationStatus = modelPreparation.data?.status || (preparePlan.isPending ? "queued" : "not_started");
  const capacityStatus = capacityCheck.data?.status || (capacity.isPending ? "queued" : "not_started");
  const preparationActive = ["queued", "running"].includes(preparationStatus)
    || ["queued", "running"].includes(capacityStatus);
  const preparationFailed = ["blocked", "failed", "cancelled"].includes(preparationStatus)
    || ["blocked", "failed", "cancelled"].includes(capacityStatus);
  return (
    <StepSurface number="07" title="Prove the training path" detail="A small deterministic run catches schema, asset, tokenizer, and memory problems before you spend the full budget.">
      {!builtVersionId ? <EmptyState icon={Database} title="Publish a dataset version first" detail="The proof run always binds an immutable version, never an editable source path." /> : (
        <div className="grid gap-px bg-border-subtle xl:grid-cols-[minmax(0,1fr)_220px]">
          <div className="space-y-5 bg-bg px-5 py-5">
            {planRecommendation.isLoading ? <ProgressStrip label="Preparing one recommended training plan" /> : planRecommendation.isError ? <ErrorBanner title="A safe plan could not be recommended" detail={(planRecommendation.error as Error).message} tone="warning" /> : planRecommendation.data ? <div className="border-l-2 border-accent bg-accent-bg/25 px-4 py-4"><div className="flex flex-wrap items-start justify-between gap-3"><div><div className="text-[10px] font-medium uppercase tracking-[0.12em] text-accent">Recommended plan</div><div className="mt-1 text-sm font-medium text-fg">{planRecommendation.data.summary}</div><div className="mt-1 text-[11px] leading-5 text-fg-muted">Update a small part of the model · process {String(planRevision?.definition.effective_batch_size || 1)} examples together · maximum text length {String(planRevision?.definition.max_sequence_length || "task default")}</div></div><Badge tone="info">{planRevision?.trainer_mode.toUpperCase()}</Badge></div><div className="mt-3 grid grid-cols-2 gap-px border-y border-border-subtle bg-border-subtle 2xl:grid-cols-4"><Readout label="Model" value={selectedModel} /><Readout label="Download" value={formatBytes(planRevision?.forecast.download_bytes)} /><Readout label="Expected memory" value={formatBytes(planRevision?.forecast.peak_memory_bytes)} sampled /><Readout label="Proof time" value={planRevision?.forecast.proof_seconds_range ? `${planRevision.forecast.proof_seconds_range[0]}–${planRevision.forecast.proof_seconds_range[1]} sec` : "Unavailable"} sampled /></div><ul className="mt-3 space-y-1.5">{planRevision?.reasons.map((reason) => <li key={reason.code} className="text-[10.5px] leading-4 text-fg-muted"><span className="font-medium text-fg">{reason.summary}.</span> {reason.detail}</li>)}</ul>{modelAccessNote ? <div className="mt-3 border-t border-border-subtle pt-3 text-[10px] leading-4 text-fg-subtle">Preparing confirms the displayed download and model access terms. {modelAccessNote}</div> : null}{planRecommendation.data.alternatives.length ? <details className="mt-3 border-t border-border-subtle pt-3"><summary className="cursor-pointer text-[10px] font-medium text-fg-subtle">Choose another compatible model</summary><div className="mt-2 space-y-2">{planRecommendation.data.alternatives.map((alternative) => <div key={alternative.model_id} className="flex items-center justify-between gap-3"><div><div className="text-[10.5px] text-fg">{alternative.label}</div><div className="text-[9.5px] text-fg-subtle">{alternative.reason_not_selected}</div></div><Button size="sm" variant="ghost" disabled={chooseAlternative.isPending} onClick={() => chooseAlternative.mutate(alternative.model_id)}>Choose</Button></div>)}</div></details> : null}</div> : null}
            {verifierRequired ? <Field label="Qualified training verifier" hint="RAFT and GRPO proof runs use an immutable candidate- or approved-qualified verifier revision."><Select value={selectedVerifierRevisionId} onValueChange={(value) => setDraft((current) => ({ ...current, verifierRevisionId: value, trainingPlanId: "", trainingPlanRevisionId: "", modelPreparationId: "", capacityCheckId: "" }))}><SelectTrigger aria-label="Qualified training verifier"><SelectValue placeholder={verifierProfiles.isLoading ? "Loading qualified verifiers" : "Choose a qualified verifier"} /></SelectTrigger><SelectContent>{qualifiedVerifiers.map(({ profile, revision }) => <SelectItem key={revision!.id} value={revision!.id}>{profile.name} · {revision!.alias}</SelectItem>)}</SelectContent></Select>{verifierProfiles.isSuccess && !qualifiedVerifiers.length ? <div className="mt-2 flex items-center justify-between gap-3 border-l-2 border-warning bg-warning-bg px-3 py-2"><p className="text-[10px] leading-4 text-fg-muted">No compatible qualified verifier is available. Calibrate and promote one before this proof run.</p><Button variant="secondary" size="sm" asChild><Link to="/eval" search={{ section: "verifiers" }}>Open Verifiers</Link></Button></div> : null}</Field> : null}
            <div className="grid grid-cols-2 gap-px border-y border-border-subtle bg-border-subtle 2xl:grid-cols-4"><Readout label="Records" value={`up to ${proofBudget}`} /><Readout label={cycleTrainer(trainerMode) ? "Cycles" : "Epochs"} value="1" /><Readout label="Seed" value="42" /><Readout label="Backend" value={backendName || "auto"} /></div>
            {preparePlan.isError ? <ErrorBanner title="Model preparation could not start" detail={(preparePlan.error as Error).message} /> : null}
            {modelPreparation.data?.error ? <ErrorBanner title="Model preparation needs attention" detail={modelPreparation.data.error} /> : null}
            {capacity.isError ? <ErrorBanner title="Capacity check could not start" detail={(capacity.error as Error).message} /> : null}
            {capacityCheck.data?.error ? <ErrorBanner title="This plan did not fit safely" detail={capacityCheck.data.primary_remedy?.reason || capacityCheck.data.error} /> : null}
            {preparationActive ? <ProgressStrip label={preparationStatus === "running" ? "Preparing the exact model" : capacityStatus === "running" ? "Measuring this training plan" : "Waiting for workstation capacity"} progress={Number(modelPreparation.data?.progress?.progress ?? capacityCheck.data?.progress?.progress ?? 0) || undefined} detail={capacityStatus === "running" ? "Halo Forge is using disposable scratch state and will remove it after the check." : "The download and capacity work are durable and can be resumed after restart."} /> : null}
            {planReady ? <div className="grid gap-px border-y border-border-subtle bg-border-subtle 2xl:grid-cols-3"><Readout label="Measured fit" value={capacityCheck.data?.status === "ready_with_adjustment" ? "Fits with safe adjustment" : "Fits as planned"} /><Readout label="Examples processed together" value={String(capacityCheck.data?.selected_adjustment?.batch_size ?? planRevision?.definition.batch_size ?? 1)} /><Readout label="Maximum text length" value={String(planRevision?.definition.max_sequence_length || "Task default")} /></div> : null}
            {planReadinessQuery.data?.notices?.map((notice) => <ErrorBanner key={notice.code} title="Separate provider check still required" detail={notice.summary} tone="warning" />)}
            {readiness.isLoading ? <ProgressStrip label="Checking dataset readiness" /> : readinessUnavailable ? <ErrorBanner title="Readiness is unavailable" detail={(readiness.error as Error)?.message || "Halo Forge could not verify this dataset and trainer combination. Retry before launching."} tone="warning" /> : blockers.length || (readiness.data?.warnings.length ?? 0) > 0 ? <ReadinessReportPanel report={readiness.data} title="Proof-run readiness" /> : planReady && preflight.isLoading ? <ProgressStrip label="Running the final proof-run preflight" /> : planReady && preflightUnavailable ? <ErrorBanner title="Model preflight is unavailable" detail={(preflight.error as Error)?.message || "Halo Forge could not verify model access and workstation resources. Nothing can launch until the check succeeds."} tone="warning" /> : preflightErrors.length ? <div className="space-y-2">{preflightErrors.map((message, index) => <ErrorBanner key={`${message}-${index}`} title={message} detail={preflight.data?.suggested_fixes?.[index] || "Resolve this item, then run model-aware preflight again."} />)}</div> : artifactPreparing ? <ProgressStrip label="Preparing the trainer-ready dataset" /> : launchReady ? <div className="border-l-2 border-success bg-success-bg px-4 py-3"><div className="text-xs font-medium text-success">Ready for a proof run</div><div className="mt-1 text-[11px] text-fg-muted">The exact model revision, managed training split, trainer adapter, and measured workstation fit are ready.</div></div> : preparationFailed ? <ErrorBanner title="Prepare and check needs attention" detail={capacityCheck.data?.primary_remedy?.label || "Review the issue above, then retry the preparation."} tone="warning" /> : !draft.modelPreparationId ? <Hint>Confirm Prepare and check to resolve the exact model revision and measure this training shape.</Hint> : null}
            {proof.isError ? <ErrorBanner title="Proof run did not start" detail={(proof.error as Error).message} /> : null}
            {draft.proofRunId ? <div className="border border-border-subtle bg-bg-subtle/45 px-4 py-3"><div className="flex flex-wrap items-center justify-between gap-3"><div><div className="text-xs font-medium text-fg">Proof run {String(proofRun.data?.status || proof.data?.status || "queued")}</div><div className="mt-1 text-[11px] text-fg-muted">The proof identity, deterministic sample, scenario, mapping, recipe, and trainer artifact are captured in replay.</div></div><Button size="sm" variant="secondary" asChild><Link to="/runs/$runId" params={{ runId: draft.proofRunId }}>Open run</Link></Button></div></div> : null}
            {outcome.data ? <div className={cn("border-l-2 px-4 py-3", outcomeSupportsFullRun ? "border-success bg-success-bg" : "border-warning bg-warning-bg")}><div className="flex flex-wrap items-start justify-between gap-3"><div><div className="text-xs font-medium text-fg">{outcomeGuidance.data?.display_status || (["queued", "running"].includes(outcome.data.status) ? "Checking training result" : "Training result ready")}</div><div className="mt-1 text-[11px] leading-5 text-fg-muted">{outcomeGuidance.data?.summary || "Halo Forge is preparing the same development evidence for the base and proof models."}</div></div>{["queued", "running"].includes(outcome.data.status) ? <Loader2 className="h-4 w-4 animate-spin text-accent" /> : !outcomeSupportsFullRun ? <Button size="sm" variant="secondary" asChild><Link to="/runs/$runId" params={{ runId: draft.proofRunId }} search={{ tab: "evaluation" }}>{outcomeGuidance.data?.primary_action.label || "Review examples"}</Link></Button> : null}</div>{!outcomeSupportsFullRun && !["queued", "running"].includes(outcome.data.status) ? <details className="mt-3 border-t border-border-subtle pt-3"><summary className="cursor-pointer text-[10px] font-medium text-fg-subtle">Continue anyway</summary><Field label="Reason" hint="Required and retained in lineage."><Input value={draft.outcomeOverrideReason || ""} onChange={(event) => setDraft((current) => ({ ...current, outcomeOverrideReason: event.target.value }))} placeholder="Why is a full run still appropriate?" /></Field></details> : null}</div> : null}
            {assessOutcome.isError ? <ErrorBanner title="Outcome assessment failed" detail={(assessOutcome.error as Error).message} /> : null}
            {full.isError ? <ErrorBanner title="Full run did not start" detail={(full.error as Error).message} /> : null}
            {draft.fullRunId ? <div className="border-l-2 border-success bg-success-bg px-4 py-3"><div className="text-xs font-medium text-success">Full run started</div><div className="mt-2"><Button size="sm" variant="secondary" asChild><Link to="/runs/$runId" params={{ runId: draft.fullRunId }}>Open full run</Link></Button></div></div> : null}
          </div>
          <aside className="bg-bg-subtle/35 px-5 py-5"><div className="text-[10px] font-medium uppercase tracking-[0.12em] text-fg-disabled">Recommended next action</div><div className="mt-3 text-sm font-medium text-fg">{selectedScenario?.label}</div><div className="mt-1 break-words text-[10px] leading-5 text-fg-subtle">{selectedModel || "Preparing a recommendation"}</div><div className="mt-5 space-y-2">{!draft.proofRunId && !planReady ? <Button variant="primary" size="lg" className="w-full" disabled={!planRevision || preparationActive || preparePlan.isPending || retryPreparation.isPending || verifierRequired && !selectedVerifierRevisionId} onClick={() => preparationFailed ? retryPreparation.mutate() : preparePlan.mutate()}>{preparationActive || preparePlan.isPending || retryPreparation.isPending ? <Loader2 className="animate-spin" /> : <ShieldCheck />}{preparationActive || retryPreparation.isPending ? "Preparing and checking" : preparationFailed ? "Retry prepare and check" : "Prepare and check"}</Button> : !draft.proofRunId ? <Button variant="primary" size="lg" className="w-full" disabled={!selectedModel || !launchReady || proof.isPending} onClick={() => proof.mutate()}>{proof.isPending ? <Loader2 className="animate-spin" /> : <Play />}Start proof run</Button> : proofComplete && !draft.outcomeAssessmentId ? <Button variant="primary" size="lg" className="w-full" disabled={assessOutcome.isPending} onClick={() => assessOutcome.mutate()}>{assessOutcome.isPending ? <Loader2 className="animate-spin" /> : <ShieldCheck />}Check training result</Button> : proofComplete && outcome.data && !draft.fullRunId && !["queued", "running"].includes(outcome.data.status) ? <Button variant={outcomeSupportsFullRun ? "primary" : "secondary"} size="lg" className="w-full" disabled={full.isPending || (!outcomeSupportsFullRun && !(draft.outcomeOverrideReason || "").trim())} onClick={() => full.mutate()}>{full.isPending ? <Loader2 className="animate-spin" /> : <ArrowRight />}{outcomeSupportsFullRun ? "Start full run" : "Continue anyway"}</Button> : <Button variant="secondary" size="lg" className="w-full" asChild><Link to="/runs/$runId" params={{ runId: draft.proofRunId }}>View progress</Link></Button>}</div><p className="mt-3 text-[10px] leading-4 text-fg-subtle">Prepare and check confirms any download, resolves the exact model revision, and measures the plan without keeping scratch weights. Training never starts automatically.</p><details className="mt-5 border-t border-border-subtle pt-4"><summary className="cursor-pointer text-[10px] font-medium text-fg-subtle">Technical details</summary><div className="mt-2 space-y-1 font-mono text-[9px] leading-4 text-fg-disabled"><div>Plan {planRevision?.status || "pending"}</div><div>Precision {String(planRevision?.definition.precision || "automatic")}</div><div>Capacity {capacityStatus.replaceAll("_", " ")}</div></div></details></aside>
        </div>
      )}
    </StepSurface>
  );
}

function StudioFooter({ context, onBack, onNext }: { context: StudioContext; onBack: () => void; onNext: () => void }) {
  const { draft, selectedScenario, inspection, inspectMutation, recipeError, buildComplete } = context;
  const index = STEPS.findIndex((step) => step.id === draft.step);
  let disabled = false;
  let label = "Continue";
  let action = onNext;
  if (draft.step === "goal") disabled = !selectedScenario;
  if (draft.step === "source") {
    disabled = !sourceReady(draft, context.selectedFiles, context.examples) || inspectMutation.isPending;
    label = inspectMutation.isPending ? "Inspecting source" : "Inspect source";
    action = () => inspectMutation.mutate();
  }
  if (draft.step === "format") disabled = !inspection || inspection.status !== "completed" || !draft.candidateConfirmed;
  if (draft.step === "map") disabled = !draft.mappingConfirmed;
  if (draft.step === "prepare") disabled = Boolean(recipeError) || draft.preparation.trainRatio + draft.preparation.validationRatio + draft.preparation.testRatio !== 100 || context.inspectionReadiness?.ready === false;
  if (draft.step === "version") {
    disabled = !buildComplete;
    label = "Continue to training";
  }
  if (draft.step === "train") return null;
  return (
    <div className="sticky bottom-0 z-10 flex min-h-16 items-center justify-between gap-3 border-t border-border bg-bg/95 px-4 py-3 backdrop-blur md:px-5">
      <Button variant="ghost" disabled={index === 0} onClick={onBack}><ArrowLeft />Back</Button>
      <div className="hidden text-[10px] text-fg-subtle sm:block">Step {index + 1} of {STEPS.length} · recommendations always require confirmation</div>
      <Button variant="primary" disabled={disabled} onClick={action}>{draft.step === "source" && inspectMutation.isPending ? <Loader2 className="animate-spin" /> : null}{label}<ArrowRight /></Button>
    </div>
  );
}

function EvidenceInspector({ step, scenario, inspection, mappingPreview, preparation, readiness, draft, buildStatus, capabilities, saveState }: { step: StudioStep; scenario?: TrainingScenarioDescriptor; inspection?: DatasetSourceInspection | null; mappingPreview?: MappingPreview | null; preparation?: DatasetPreparationPlan | null; readiness?: DatasetReadiness | null; draft: OwnDataDraft; buildStatus?: string; capabilities: Array<{ id: string; kind: string; label: string; execution_surface?: string | null; available: boolean; status?: string; reason?: string | null }>; saveState: string }) {
  return (
    <aside className="bg-bg-subtle/35">
      <div className="sticky top-0">
        <div className="flex items-center justify-between border-b border-border-subtle px-4 py-3"><div><div className="text-[9.5px] font-medium uppercase tracking-[0.13em] text-fg-disabled">Inspector</div><div className="mt-0.5 text-xs font-medium text-fg">{STEPS.find((item) => item.id === step)?.label} evidence</div></div><span className="font-mono text-[9.5px] text-fg-subtle">{saveState}</span></div>
        <dl className="divide-y divide-border-subtle">
          <InspectorRow label="Scenario" value={scenario?.label || "Not selected"} />
          <InspectorRow label="Training shape" value={scenario ? scenarioKindLabel(scenario) : "—"} />
          {inspection ? <InspectorRow label={isCorpusScenario(scenario) ? "Documents" : "Scanned records"} value={formatInteger(extractionSummary(inspection).documentCount ?? inspection.row_count)} mono /> : null}
          {mappingPreview ? <><InspectorRow label="Preview valid" value={`${mappingPreview.valid_count}/${mappingPreview.total_sampled}`} mono /><InspectorRow label="Preview status" value={mappingPreview.ready ? "Ready" : "Needs attention"} /></> : null}
          {preparation ? <InspectorRow label="Recipe steps" value={String(preparation.recipe.steps.length)} mono /> : null}
          {readiness ? <InspectorRow label="Readiness" value={readiness.ready ? "Ready" : `${readiness.blockers.length} item${readiness.blockers.length === 1 ? "" : "s"} to resolve`} /> : null}
          {draft.buildJobId ? <InspectorRow label="Build" value={buildStatus || "queued"} /> : null}
        </dl>
        {inspection?.source_fingerprint ? <details className="border-b border-border-subtle px-4 py-4"><summary className="cursor-pointer text-[10px] font-medium text-fg-subtle">Advanced · technical provenance</summary><div className="mt-3"><InspectorRow label="Source fingerprint" value={shortHash(inspection.source_fingerprint)} mono /></div></details> : null}
        {inspection?.fields.length ? <div className="border-b border-border-subtle px-4 py-4"><div className="text-[9.5px] font-medium uppercase tracking-[0.12em] text-fg-disabled">Detected fields</div><div className="mt-2 space-y-2">{inspection.fields.slice(0, 10).map((field) => <div key={field.name} className="flex items-center justify-between gap-3 text-[10px]"><span className="truncate font-mono text-fg-muted">{field.name}</span><span className="font-mono text-fg-subtle">{percent(field.coverage)}</span></div>)}</div></div> : null}
        {inspection?.parse_errors?.length ? <details className="border-b border-border-subtle px-4 py-4"><summary className="cursor-pointer text-[10px] font-medium text-warning">Advanced · parse issues ({inspection.parse_errors.length})</summary><pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap font-mono text-[9px] leading-4 text-fg-subtle">{JSON.stringify(inspection.parse_errors.slice(0, 5), null, 2)}</pre></details> : null}
        {step === "source" ? <div className="border-b border-border-subtle px-4 py-4"><div className="text-[9.5px] font-medium uppercase tracking-[0.12em] text-fg-disabled">Supported presentations</div><div className="mt-2 space-y-2">{capabilities.filter((item) => item.kind === "execution_surface").map((item) => <div key={item.id} className="border-b border-border-subtle/60 pb-2 last:border-0 last:pb-0"><div className="flex items-center justify-between gap-3 text-[10px]"><span className="text-fg-muted">{item.label}</span><Badge size="sm" tone={item.available ? "success" : "neutral"}>{item.status || (item.available ? "available" : "unavailable")}</Badge></div>{item.reason ? <p className="mt-1 text-[9px] leading-4 text-fg-disabled">{item.reason}</p> : null}</div>)}</div></div> : null}
        <div className="px-4 py-4"><div className="flex gap-2 text-[10px] leading-4 text-fg-subtle"><ShieldCheck className="mt-0.5 h-3.5 w-3.5 shrink-0 text-accent" /><span>Source files are never modified. Building, publishing, and training are separate explicit actions.</span></div></div>
      </div>
    </aside>
  );
}

function StepSurface({ number, title, detail, action, children }: { number: string; title: string; detail: string; action?: ReactNode; children: ReactNode }) {
  return <section><header className="flex flex-wrap items-start justify-between gap-3 border-b border-border-subtle px-5 py-5"><div className="flex min-w-0 gap-3"><span className="mt-0.5 font-mono text-[10px] text-accent">{number}</span><div><h2 className="text-base font-semibold text-fg">{title}</h2><p className="mt-1 max-w-2xl text-[11px] leading-5 text-fg-muted">{detail}</p></div></div>{action}</header>{children}</section>;
}

function SourceModeButton({ icon: Icon, label, detail, active, onClick }: { icon: LucideIcon; label: string; detail: string; active: boolean; onClick: () => void }) {
  return <button type="button" onClick={onClick} aria-pressed={active} className={cn("min-h-24 bg-bg px-4 py-4 text-left transition-colors", active ? "bg-accent-bg/55" : "hover:bg-surface")}><Icon className={cn("h-4 w-4", active ? "text-accent" : "text-fg-subtle")} /><span className={cn("mt-2 block text-[12px] font-medium", active ? "text-accent" : "text-fg")}>{label}</span><span className="mt-0.5 block text-[10px] leading-4 text-fg-subtle">{detail}</span></button>;
}

function CorpusSourceFlow({ extractors, loading }: { extractors?: Array<{ label: string; available: boolean; extensions?: string[]; media_types?: string[]; reason?: string | null }>; loading: boolean }) {
  const available = extractors?.filter((item) => item.available) ?? [];
  const extensions = Array.from(new Set(available.flatMap((item) => item.extensions ?? []))).slice(0, 8);
  return (
    <div className="border-b border-border-subtle bg-bg-subtle/35">
      <div className="grid gap-px bg-border-subtle sm:grid-cols-3">
        {[
          { icon: FolderOpen, label: "Source", detail: "Files, a folder, a pinned dataset, or a verified example." },
          { icon: FileText, label: "Extraction", detail: "Visible text, headings, code, and document provenance are retained." },
          { icon: ListChecks, label: "Preparation", detail: "Failures, duplicates, and split boundaries are reviewed before packing." },
        ].map(({ icon: Icon, label, detail }, index) => <div key={label} className="bg-bg px-4 py-3"><div className="flex items-center gap-2"><span className="font-mono text-[9px] text-accent">{String(index + 1).padStart(2, "0")}</span><Icon className="h-3.5 w-3.5 text-fg-subtle" /><span className="text-[11px] font-medium text-fg">{label}</span></div><p className="mt-1.5 text-[10px] leading-4 text-fg-subtle">{detail}</p></div>)}
      </div>
      <div className="flex flex-wrap items-center gap-2 px-4 py-2.5 text-[10px] text-fg-subtle">
        <span>{loading ? "Checking installed document extractors…" : available.length ? `${available.length} document extractor${available.length === 1 ? "" : "s"} available` : "Reviewed extraction support will be verified during inspection"}</span>
        {(extensions.length ? extensions : [".txt", ".md", ".html", ".pdf", ".docx"]).map((extension) => <span key={extension} className="rounded-sm border border-border px-1.5 py-0.5 font-mono text-[9px]">{extension.startsWith(".") ? extension : `.${extension}`}</span>)}
      </div>
    </div>
  );
}

function CorpusExtractionReview({ inspection }: { inspection: DatasetSourceInspection }) {
  const summary = extractionSummary(inspection);
  const failures = (summary.failed ?? 0) + (summary.quarantined ?? 0);
  return (
    <div className="border-b border-border-subtle bg-bg-subtle/35 px-5 py-4">
      <div className="grid gap-4 sm:grid-cols-3">
        <div><div className="flex items-center gap-2 text-[11px] font-medium text-fg"><FileText className="h-3.5 w-3.5 text-accent" />Visible content</div><p className="mt-1 text-[10px] leading-4 text-fg-subtle">Text, headings, lists, tables, and code are normalized into document records.</p></div>
        <div><div className="flex items-center gap-2 text-[11px] font-medium text-fg"><Layers3 className="h-3.5 w-3.5 text-accent" />Document boundaries</div><p className="mt-1 text-[10px] leading-4 text-fg-subtle">Each extracted document keeps its source reference and spans for grouped splitting.</p></div>
        <div><div className="flex items-center gap-2 text-[11px] font-medium text-fg"><ShieldCheck className="h-3.5 w-3.5 text-accent" />Failure handling</div><p className="mt-1 text-[10px] leading-4 text-fg-subtle">Encrypted, image-only, empty, and unsupported files are quarantined rather than silently dropped.</p></div>
      </div>
      {failures ? <div className="mt-4 border-l-2 border-warning bg-warning-bg px-3 py-2 text-[10px] leading-4 text-fg-muted">{formatInteger(failures)} document{failures === 1 ? "" : "s"} need extraction review. You can inspect representative failures in readiness before publication.</div> : null}
    </div>
  );
}

function ReadinessReportPanel({ report, loading = false, title, onAction }: { report?: DatasetReadiness | null; loading?: boolean; title: string; onAction?: (target: string) => void }) {
  if (loading) return <div className="border-t border-border-subtle px-5 pb-5"><ProgressStrip label="Analyzing readiness and remediation" /></div>;
  if (!report) return null;
  const findings = [
    ...report.blockers.map((finding) => ({ ...finding, tone: "danger" as const })),
    ...report.warnings.map((finding) => ({ ...finding, tone: "warning" as const })),
  ];
  const actions = report.actions ?? [];
  const summary = report.summary ?? {};
  return (
    <section className="border-t border-border-subtle" aria-label={title}>
      <div className="flex flex-wrap items-start justify-between gap-3 px-5 py-4">
        <div><div className="flex items-center gap-2 text-xs font-medium text-fg">{report.ready ? <CheckCircle2 className="h-4 w-4 text-success" /> : <ListChecks className="h-4 w-4 text-warning" />}{title}</div><div className="mt-1 text-[11px] leading-5 text-fg-muted">{readinessSummary(report)}</div></div>
        <Badge size="sm" tone={report.ready ? "success" : "warning"}>{report.ready ? "Ready" : `${report.blockers.length} to resolve`}</Badge>
      </div>
      {typeof summary.estimated_accepted_records === "number" || typeof summary.estimated_quarantined_records === "number" ? <div className="grid gap-px border-y border-border-subtle bg-border-subtle sm:grid-cols-3"><Readout label="Estimated accepted" value={formatInteger(summary.estimated_accepted_records)} sampled={report.sampled} /><Readout label="Estimated quarantined" value={formatInteger(summary.estimated_quarantined_records)} sampled={report.sampled} /><Readout label="Preview duplicates" value={formatInteger(summary.exact_duplicate_preview_records)} sampled={report.sampled} /></div> : null}
      {findings.length ? <div className="divide-y divide-border-subtle border-b border-border-subtle">{findings.map((finding, index) => {
        const action = actions.find((candidate) => candidate.id === finding.action_id);
        return <div key={`${finding.code || finding.message}-${index}`} className="grid gap-3 px-5 py-3 sm:grid-cols-[minmax(0,1fr)_auto]"><div><div className={cn("text-[11px] font-medium", finding.tone === "danger" ? "text-danger" : "text-warning")}>{finding.message}</div>{finding.why_it_matters ? <div className="mt-1 text-[10px] leading-4 text-fg-subtle">{finding.why_it_matters}</div> : null}{finding.remedy ? <div className="mt-1 text-[10px] leading-4 text-fg-muted">{finding.remedy}</div> : null}</div>{action && onAction ? <Button size="sm" variant="secondary" onClick={() => onAction(action.target || action.action)}>{action.label}</Button> : null}</div>;
      })}</div> : null}
      {actions.length ? <div className="px-5 py-4"><div className="text-[9.5px] font-medium uppercase tracking-[0.12em] text-fg-disabled">Recommended remediation</div><ol className="mt-3 space-y-3">{actions.map((action, index) => <li key={action.id} className="grid grid-cols-[20px_minmax(0,1fr)] gap-2 text-[10px] leading-4"><span className="font-mono text-accent">{String(index + 1).padStart(2, "0")}</span><span><span className="font-medium text-fg">{action.label}</span><span className="mt-0.5 block text-fg-subtle">{action.description}</span></span></li>)}</ol></div> : report.ready ? <div className="border-t border-border-subtle px-5 py-3 text-[10px] leading-4 text-fg-subtle">No blocking remediation is required. Quality sufficiency remains a separate experimental judgment.</div> : null}
    </section>
  );
}

function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return <label className="block space-y-1.5"><span className="block text-[11px] font-medium text-fg-muted">{label}</span>{children}{hint ? <span className="block text-[10px] leading-4 text-fg-subtle">{hint}</span> : null}</label>;
}

function PreparationToggle({ index, title, detail, checked, onChange }: { index: string; title: string; detail: string; checked: boolean; onChange: (value: boolean) => void }) {
  return <label className="grid cursor-pointer gap-3 px-5 py-4 transition-colors hover:bg-surface/35 sm:grid-cols-[minmax(0,1fr)_auto]"><span><span className="flex items-center gap-2 text-[12px] font-medium text-fg"><span className="font-mono text-[9px] text-fg-disabled">{index}</span>{title}</span><span className="mt-1 block text-[11px] leading-5 text-fg-muted">{detail}</span></span><input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} className="mt-1 h-4 w-4 accent-accent" /></label>;
}

function Readout({ label, value, sampled = false }: { label: string; value: string; sampled?: boolean }) {
  return <div className="bg-bg px-4 py-3"><div className="font-mono text-lg text-fg">{value}</div><div className="mt-1 text-[9px] uppercase tracking-[0.12em] text-fg-disabled">{label}{sampled ? " · sampled" : ""}</div></div>;
}

function SummaryRow({ label, value }: { label: string; value: string }) { return <div className="grid grid-cols-[130px_minmax(0,1fr)] gap-4 px-3 py-2.5 text-[11px]"><span className="text-fg-subtle">{label}</span><span className="text-fg">{value}</span></div>; }
function InspectorRow({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) { return <div className="grid grid-cols-[105px_minmax(0,1fr)] gap-3 px-4 py-2.5 text-[10px]"><dt className="text-fg-subtle">{label}</dt><dd className={cn("truncate text-right text-fg-muted", mono && "font-mono")} title={value}>{value}</dd></div>; }
function PreviewObject({ label, value }: { label: string; value: Record<string, unknown> }) { return <div className="px-4 py-3"><div className="mb-2 text-[9px] font-medium uppercase tracking-[0.12em] text-fg-disabled">{label}</div><pre className="max-h-48 overflow-auto whitespace-pre-wrap break-words font-mono text-[10px] leading-5 text-fg-muted">{JSON.stringify(value, null, 2)}</pre></div>; }
function SelectedSource({ value, detail }: { value: string; detail: string }) { return <div className="border-l-2 border-success bg-success-bg px-4 py-3"><div className="break-all font-mono text-[11px] text-fg">{value}</div><div className="mt-1 text-[10px] text-fg-muted">{detail}</div></div>; }
function Hint({ children }: { children: ReactNode }) { return <div className="border-l border-border-strong pl-3 text-[11px] leading-5 text-fg-muted">{children}</div>; }
function FileList({ files }: { files: File[] }) { return <div className="max-h-40 divide-y divide-border-subtle overflow-auto border-y border-border-subtle">{files.slice(0, 100).map((file) => <div key={`${file.webkitRelativePath}-${file.name}-${file.size}`} className="flex items-center justify-between gap-3 px-3 py-2 text-[10px]"><span className="truncate font-mono text-fg-muted">{file.webkitRelativePath || file.name}</span><span className="shrink-0 font-mono text-fg-subtle">{formatBytes(file.size)}</span></div>)}</div>; }

function ProgressStrip({ label, progress, detail }: { label: string; progress?: number; detail?: string }) { return <div className="mt-4 border border-border-subtle bg-bg-subtle px-3 py-3"><div className="flex items-center justify-between gap-3 text-[11px]"><span className="flex items-center gap-2 font-medium text-fg"><Loader2 className="h-3.5 w-3.5 animate-spin text-accent" />{label}</span><span className="font-mono text-fg-subtle">{progress !== undefined ? `${Math.round(progress)}%` : "working"}</span></div><div className="mt-2 h-1 overflow-hidden bg-surface"><div className={cn("h-full bg-accent transition-all", progress === undefined && "w-1/3 animate-pulse")} style={progress !== undefined ? { width: `${Math.max(1, Math.min(100, progress))}%` } : undefined} /></div>{detail ? <div className="mt-2 text-[10px] text-danger">{detail}</div> : null}</div>; }
function ErrorBanner({ title, detail, tone = "danger" }: { title: string; detail: string; tone?: "danger" | "warning" }) { return <div className={cn("border-l-2 px-4 py-3", tone === "danger" ? "border-danger bg-danger-bg" : "border-warning bg-warning-bg")}><div className={cn("text-xs font-medium", tone === "danger" ? "text-danger" : "text-warning")}>{title}</div><div className="mt-1 text-[11px] leading-5 text-fg-muted">{detail}</div></div>; }
function LoadingState({ label, progress }: { label: string; progress?: number | null }) { return <div className="flex min-h-52 flex-col items-center justify-center px-6 py-14 text-center"><Loader2 className="h-6 w-6 animate-spin text-accent" /><div className="mt-3 text-xs font-medium text-fg">{label}</div>{progress !== undefined && progress !== null ? <div className="mt-1 font-mono text-[10px] text-fg-subtle">{Math.round(progress)}%</div> : null}</div>; }
function ErrorState({ label }: { label: string }) { return <div className="flex min-h-52 flex-col items-center justify-center px-6 py-14 text-center"><XCircle className="h-6 w-6 text-danger" /><div className="mt-3 max-w-lg text-xs text-fg-muted">{label}</div></div>; }
function EmptyState({ icon: Icon, title, detail, compact = false }: { icon: LucideIcon; title: string; detail: string; compact?: boolean }) { return <div className={cn("flex flex-col items-center justify-center px-6 text-center", compact ? "py-8" : "min-h-52 py-14")}><Icon className="h-6 w-6 text-fg-disabled" /><div className="mt-3 text-xs font-medium text-fg">{title}</div><div className="mt-1 max-w-md text-[11px] leading-5 text-fg-muted">{detail}</div></div>; }

function RestoreDraftBanner({ name, onRestore, onDiscard }: { name: string; onRestore: () => void; onDiscard: () => void }) { return <div className="flex flex-wrap items-center justify-between gap-3 border-b border-accent/30 bg-accent-bg px-5 py-3"><div className="flex items-center gap-2 text-[11px] text-fg"><RotateCcw className="h-3.5 w-3.5 text-accent" /><span>A saved “{name}” workflow is available.</span></div><div className="flex gap-2"><Button variant="ghost" size="sm" onClick={onDiscard}>Discard</Button><Button variant="primary" size="sm" onClick={onRestore}>Restore draft</Button></div></div>; }
function ConfidenceBadge({ confidence }: { confidence: string }) { return <Badge size="sm" tone={confidence === "high" ? "success" : confidence === "medium" ? "warning" : "neutral"}>{confidence} confidence</Badge>; }

function withCorpusScenarioFallback(scenarios: TrainingScenarioDescriptor[]): TrainingScenarioDescriptor[] {
  if (scenarios.some((scenario) => scenario.id === "corpus-adaptation" || scenario.revision_id === "corpus-adaptation@1")) return scenarios;
  return [...scenarios, {
    id: "corpus-adaptation",
    revision_id: "corpus-adaptation@1",
    revision: 1,
    label: "Adapt a model to documents",
    description: "Continue language-model training on a reviewed collection of documents while preserving source provenance.",
    modality: "text",
    canonical_shape: "corpus",
    task_type: "continued_pretraining",
    available: true,
    verified: true,
    required_fields: [
      { name: "document_id", label: "Document identity", description: "Stable logical record identity; content changes are tracked separately.", required: true },
      { name: "document_hash", label: "Document fingerprint", description: "Content identity used for duplicate checks.", required: true },
      { name: "text", label: "Extracted text", description: "Visible document content.", required: true },
      { name: "source_ref", label: "Source reference", description: "Original file or row reference retained for provenance and boundary-safe splitting.", required: true },
    ],
    optional_fields: [
      { name: "title", label: "Title" },
      { name: "source_spans", label: "Source spans" },
      { name: "timestamp", label: "Timestamp" },
      { name: "metadata", label: "Metadata" },
    ],
    accepted_aliases: {
      document_id: ["document_id", "id", "doc_id"],
      document_hash: ["document_hash", "content_hash", "hash"],
      text: ["text", "content", "body", "document"],
      source_ref: ["source_ref", "source", "path", "filename", "url"],
    },
    source_layouts: ["txt", "markdown", "html", "pdf", "docx", "document_directory", "jsonl", "huggingface"],
    trainer_modes: ["cpt"],
    model_families: ["qwen2.5", "qwen2", "llama-3", "mistral"],
    default_recipe: { name: "guided-corpus-adaptation", schema: "corpus", seed: 42, steps: [] },
    common_failures: ["A document is encrypted, empty, image-only, or unsupported", "Near-duplicate documents dominate the corpus"],
    example_count: 1,
  }];
}

function fallbackGuidedExamples(scenarios: TrainingScenarioDescriptor[]): GuidedExampleDescriptor[] {
  const known: Record<string, { id: string; label: string; format: string; filename: string; records: number }> = {
    "instruction-sft": { id: "instruction-text", label: "Question and answer", format: "jsonl", filename: "instruction-text.jsonl", records: 2 },
    "chat-sft": { id: "sharegpt-chat", label: "Multi-turn support chat", format: "jsonl", filename: "sharegpt-chat.jsonl", records: 2 },
    "preference-pairs": { id: "preference-basic", label: "Chosen and rejected responses", format: "jsonl", filename: "preference-basic.jsonl", records: 2 },
    "prompt-reward": { id: "reward-prompts", label: "Verifier-scored prompts", format: "jsonl", filename: "reward-prompts.jsonl", records: 2 },
    "reasoning-sft": { id: "reasoning-worked", label: "Worked solutions", format: "jsonl", filename: "reasoning-worked.jsonl", records: 2 },
    "tool-agentic": { id: "tool-trace", label: "Tool call trace", format: "jsonl", filename: "tool-trace.jsonl", records: 1 },
    "vlm-captioning": { id: "vlm-captions", label: "Image captions", format: "jsonl", filename: "vlm-captions.jsonl", records: 1 },
    "vlm-qa": { id: "vlm-question-answer", label: "Visual question answering", format: "jsonl", filename: "vlm-question-answer.jsonl", records: 1 },
    "audio-asr": { id: "audio-transcripts", label: "Audio transcripts", format: "jsonl", filename: "audio-transcripts.jsonl", records: 1 },
    "corpus-adaptation": { id: "corpus-markdown", label: "Small document collection", format: "markdown", filename: "corpus-notes.md", records: 2 },
  };
  return scenarios.flatMap((scenario) => {
    const example = known[scenario.id];
    if (!example) return [];
    return [{
      id: example.id,
      scenario_id: scenario.id,
      scenario_revision_id: scenario.revision_id,
      label: example.label,
      description: scenario.description,
      expected_source_shape: scenarioFields(scenario, true).map((field) => field.label || humanize(field.name)).join(", "),
      expected_outcome: expectedScenarioOutcome(scenario),
      hardware_guidance: isCorpusScenario(scenario) ? "Sequence length and adaptation method determine memory use." : "Halo Forge recommends the smallest verified model that fits the active workstation.",
      fixture_format: example.format,
      fixture_filename: example.filename,
      record_count: example.records,
      modality: scenario.modality,
      trainer_modes: scenario.trainer_modes ?? [preferredTrainerMode(scenario)],
      documentation_anchor: scenario.documentation_anchor || "",
    }];
  });
}

function examplesFromGallery(examples: GuidedExampleDescriptor[], scenarioRevisionId?: string): TrainingScenarioExample[] {
  return examples
    .filter((example) => !scenarioRevisionId || example.scenario_revision_id === scenarioRevisionId)
    .map((example) => ({
      id: example.id,
      scenario_revision_id: example.scenario_revision_id,
      label: example.label,
      description: example.description,
      format: example.fixture_format,
      filename: example.fixture_filename,
      size_bytes: null,
    }));
}

function localScenarioAdvice(request: ScenarioAdviceRequest, scenarios: TrainingScenarioDescriptor[]): ScenarioAdviceResult {
  const goal = request.goal.toLowerCase();
  const terms: Record<string, string[]> = {
    "instruction-sft": ["answer", "instruction", "question", "code", "completion"],
    "chat-sft": ["chat", "conversation", "dialogue", "messages"],
    "preference-pairs": ["preference", "chosen", "rejected", "rank", "better answer"],
    "prompt-reward": ["reward", "verifier", "prompt", "raft", "grpo"],
    "reasoning-sft": ["reasoning", "worked", "solution", "steps"],
    "tool-agentic": ["tool", "function", "agent", "trace"],
    "vlm-captioning": ["image", "caption", "photo", "describe"],
    "vlm-qa": ["visual", "image question", "ocr", "invoice"],
    "audio-asr": ["audio", "speech", "transcript", "transcribe"],
    "corpus-adaptation": ["document", "corpus", "domain language", "pretrain", "adapt", "pdf", "manual", "prose"],
  };
  const recommendations = scenarios.map((scenario) => {
    const matches = (terms[scenario.id] ?? []).filter((term) => goal.includes(term));
    let score = matches.length ? Math.min(0.9, 0.38 + matches.length * 0.13) : 0.05;
    const cautions: string[] = [];
    if (request.modality) {
      if (request.modality === scenario.modality || (request.modality === "text" && isCorpusScenario(scenario))) score += 0.12;
      else { score -= 0.25; cautions.push(`This path expects ${scenario.modality} source data.`); }
    }
    if (request.source_layout && scenario.source_layouts?.includes(request.source_layout)) score += 0.14;
    return {
      scenario_id: scenario.id,
      scenario_revision_id: scenario.revision_id,
      label: scenario.label,
      score: Math.max(0, Math.min(1, score)),
      confidence: score >= 0.72 ? "high" : score >= 0.46 ? "medium" : "low",
      why_fit: matches.length ? [`Your goal mentions ${matches.slice(0, 3).join(", ")}.`] : ["Review the expected source shape to confirm this path."],
      cautions,
      required_fields: scenarioFields(scenario, true).map((field) => field.name),
      optional_fields: scenarioFields(scenario, false).map((field) => field.name),
      trainer_modes: scenario.trainer_modes,
      expected_outcome: expectedScenarioOutcome(scenario),
      available: scenario.available,
      unavailable_reason: scenario.unavailable_reason,
      requires_confirmation: true,
    };
  }).sort((left, right) => right.score - left.score);
  return {
    recommendations: recommendations.filter((item) => item.available),
    unavailable: recommendations.filter((item) => !item.available),
    explanation: "Local guidance is shown because the advisor route is unavailable. It uses the stated goal, modality, and source layout; you still confirm the scenario after inspection.",
    requires_confirmation: true,
  };
}

function buildLocalSemanticPreview(preview?: MappingPreview | null, scenario?: TrainingScenarioDescriptor): SemanticPreviewResponse | null {
  if (!preview) return null;
  const kind = scenario?.canonical_shape || "sft";
  const items = preview.items.map((item) => {
    const canonical = item.canonical;
    let title = "Training record";
    let summary = "Review the canonical values the trainer will receive.";
    let presentation: SemanticRecordPreview["presentation"] = {};
    if (kind === "chat" || kind === "tool") {
      const turns = Array.isArray(canonical.messages) ? canonical.messages.flatMap((value, index) => {
        if (!value || typeof value !== "object") return [];
        const message = value as Record<string, unknown>;
        return [{ index, role: String(message.role || message.from || "unknown"), content: String(message.content || message.value || ""), tool_calls: Array.isArray(message.tool_calls) ? message.tool_calls : [] }];
      }) : [];
      title = kind === "tool" ? "Tool trace" : "Conversation";
      summary = `${turns.length} ordered turn${turns.length === 1 ? "" : "s"}`;
      presentation = { turns, tools: asArray(canonical.tools), expected_calls: asArray(canonical.expected_calls), expected_results: asArray(canonical.expected_results) };
    } else if (kind === "preference") {
      title = String(canonical.prompt || "Preference pair").slice(0, 80);
      summary = "Compare the reviewed preferred and non-preferred responses.";
      presentation = { prompt: canonical.prompt, chosen: canonical.chosen, rejected: canonical.rejected, system: canonical.system };
    } else if (kind === "vlm") {
      title = String(canonical.prompt || "Image record").slice(0, 80);
      summary = String(canonical.response || canonical.ground_truth || "Image and text example").slice(0, 160);
      presentation = { image: canonical.image, prompt: canonical.prompt, response: canonical.response, ground_truth: canonical.ground_truth, alternatives: asArray(canonical.alternatives) };
    } else if (kind === "audio") {
      title = String(canonical.transcript || canonical.label || "Audio record").slice(0, 80);
      summary = humanize(String(canonical.task || "audio"));
      presentation = { audio: canonical.audio, task: canonical.task, transcript: canonical.transcript, label: canonical.label, metadata: asRecord(canonical.metadata) };
    } else if (kind === "corpus") {
      const text = String(canonical.text || "");
      title = String(canonical.title || displayReference(canonical.source_ref) || "Corpus document").slice(0, 80);
      summary = `${formatInteger(text.length)} characters · ${text ? text.split(/\n+/).length : 0} paragraphs`;
      presentation = { title: canonical.title, text, source_ref: canonical.source_ref, source_spans: asArray(canonical.source_spans), metadata: asRecord(canonical.metadata) };
    } else {
      title = String(canonical.prompt || "Training record").slice(0, 80);
      summary = String(canonical.response || canonical.reference_answer || canonical.text || "").slice(0, 160);
      presentation = { system: canonical.system, prompt: canonical.prompt, response: canonical.response, reference_answer: canonical.reference_answer, text: canonical.text, metadata: asRecord(canonical.metadata) };
    }
    return { kind, ordinal: item.ordinal, title, summary, source: item.source, canonical, presentation, issues: item.issues, provenance: {} };
  });
  return { items, total: items.length, limit: items.length, offset: 0, canonical_schema: kind, sampled: true };
}

function buildLocalInspectionReadiness(inspection?: DatasetSourceInspection | null, preview?: MappingPreview | null, preparation?: DatasetPreparationPlan | null, scenario?: TrainingScenarioDescriptor): DatasetReadiness | null {
  if (!inspection || !preview || !preparation) return null;
  const total = inspection.row_count ?? preview.total_sampled;
  const ratio = preview.total_sampled ? preview.valid_count / preview.total_sampled : 0;
  const accepted = Math.round(total * ratio);
  const minimum = isCorpusScenario(scenario) ? 2 : 10;
  const blockers: DatasetReadiness["blockers"] = [];
  const warnings: DatasetReadiness["warnings"] = [];
  const actions: NonNullable<DatasetReadiness["actions"]> = [];
  if (!preview.ready || preview.valid_count === 0) {
    blockers.push({ code: "no_valid_records", message: "No retained preview record satisfies the confirmed mapping.", severity: "error", action_id: "review_mapping" });
    actions.push({ id: "review_mapping", label: "Review field mapping", action: "open_mapping", description: "Return to Map and connect every required field.", target: "map" });
  }
  if (preview.invalid_count > 0) {
    blockers.push({ code: "sample_mapping_errors", message: `${preview.invalid_count} of ${preview.total_sampled} preview records need mapping or validation review.`, severity: "error", action_id: "inspect_rejected" });
    actions.push({ id: "inspect_rejected", label: "Inspect rejected examples", action: "open_rejected_records", description: "Review representative failures and adjust mapping or quarantine rules.", target: "format" });
  }
  if (accepted < minimum) {
    blockers.push({ code: "insufficient_records_for_split", message: `About ${formatInteger(accepted)} accepted records are expected; at least ${minimum} are needed for the reviewed split.`, severity: "error", action_id: "add_source_records" });
    actions.push({ id: "add_source_records", label: "Add more source data", action: "open_source", description: "Select a larger source or add files before publishing.", target: "source" });
  }
  const extraction = extractionSummary(inspection);
  if ((extraction.failed ?? 0) + (extraction.quarantined ?? 0) > 0) {
    warnings.push({ code: "document_extraction_failures", message: `${formatInteger((extraction.failed ?? 0) + (extraction.quarantined ?? 0))} documents could not be extracted and will be quarantined.`, severity: "warning", action_id: "inspect_extraction_failures" });
    actions.push({ id: "inspect_extraction_failures", label: "Inspect extraction failures", action: "open_extraction_failures", description: "Review encrypted, image-only, empty, or unsupported documents.", target: "format" });
  }
  const split = preparation.recipe.steps.find((step) => step.kind === "split");
  const ratios = (split?.ratios && typeof split.ratios === "object" ? split.ratios : {}) as Record<string, number>;
  return {
    scope: "inspection",
    ready: blockers.length === 0,
    sampled: true,
    scenario_revision_id: scenario?.revision_id,
    blockers,
    warnings,
    actions,
    summary: {
      source_records: total,
      preview_records: preview.total_sampled,
      valid_preview_records: preview.valid_count,
      invalid_preview_records: preview.invalid_count,
      estimated_accepted_records: accepted,
      estimated_quarantined_records: Math.max(0, total - accepted),
      exact_duplicate_preview_records: preparation.estimates?.duplicates,
      token_count_is_estimated: true,
    },
    split_balance: Object.fromEntries(Object.entries(ratios).map(([name, value]) => [name, { ratio: value, estimated_records: Math.round(accepted * value) }])),
    extraction: inspection.extraction_summary ?? inspection.statistics?.extraction_summary ?? {},
    minimum_data: {
      required_for_default_split: minimum,
      estimated_available: accepted,
      satisfied: accepted >= minimum,
      scientific_quality_threshold: null,
      note: "This is an operational split minimum, not a claim that the corpus is sufficient for a useful model.",
    },
  };
}

function readinessSummary(report: DatasetReadiness): string {
  if (report.summary?.headline) return report.summary.headline;
  if (report.ready) return report.warnings.length ? `Operational checks pass with ${report.warnings.length} warning${report.warnings.length === 1 ? "" : "s"} to review.` : "Operational checks pass for the selected preparation and training path.";
  return report.blockers[0]?.message || "Resolve the remaining readiness findings before continuing.";
}

function remediationTargetStep(target: string): StudioStep | null {
  if (target === "source" || target.includes("source")) return "source";
  if (target === "format" || target.includes("rejected") || target.includes("extraction")) return "format";
  if (target === "map" || target.includes("mapping") || target.includes("media")) return "map";
  if (target.includes("dedup") || target.includes("preparation")) return "prepare";
  return null;
}

function preparationDefaults(scenario?: TrainingScenarioDescriptor): PreparationSettings {
  return { ...(isCorpusScenario(scenario) ? CORPUS_PREPARATION : DEFAULT_PREPARATION) };
}

function isCorpusScenario(scenario?: TrainingScenarioDescriptor): boolean {
  return Boolean(scenario && (scenario.id === "corpus-adaptation" || scenario.canonical_shape === "corpus" || scenario.task_type === "continued_pretraining"));
}

function scenarioKindLabel(scenario: TrainingScenarioDescriptor): string {
  if (isCorpusScenario(scenario)) return "Document corpus";
  const labels: Record<string, string> = {
    sft: "Instruction examples",
    chat: "Conversation",
    preference: "Preference pairs",
    rlvr: "Verifier-guided prompts",
    tool: "Tool traces",
    vlm: "Image + text",
    audio: "Audio + text",
  };
  return labels[scenario.canonical_shape] || humanize(scenario.task_type || scenario.canonical_shape);
}

function expectedScenarioOutcome(scenario: TrainingScenarioDescriptor): string {
  if (isCorpusScenario(scenario)) return "A causal language model adapted to the language and structure of the document corpus.";
  const outcomes: Record<string, string> = {
    chat: "A model that continues conversations in the demonstrated roles and style.",
    preference: "A model optimized toward the reviewed response preferences.",
    tool: "A model that emits the demonstrated tool calls and result traces.",
    vlm: "A vision-language model adapted to the reviewed image-and-text task.",
    audio: "An audio model adapted to the reviewed transcripts.",
    sft: "A model that follows the demonstrated instructions and answers.",
  };
  return outcomes[scenario.canonical_shape] || scenario.description;
}

function extractionSummary(inspection: DatasetSourceInspection): { documentCount?: number; extracted?: number; failed?: number; quarantined?: number } {
  const value = inspection.extraction_summary ?? inspection.statistics?.extraction_summary ?? {};
  return {
    documentCount: numberValue(value.document_count) ?? inspection.row_count ?? undefined,
    extracted: numberValue(value.extracted) ?? inspection.valid_records ?? undefined,
    failed: numberValue(value.failed) ?? numberValue(value.empty) ?? 0,
    quarantined: numberValue(value.quarantined) ?? inspection.invalid_records ?? 0,
  };
}

function localCorpusProfile(inspection?: DatasetSourceInspection | null, preparation?: DatasetPreparationPlan | null): CorpusProfile {
  const records = inspection?.preview_records ?? [];
  const texts = records.map((record) => String(record.text || record.content || record.body || "")).filter(Boolean);
  const characterCount = texts.reduce((sum, text) => sum + text.length, 0);
  return {
    document_count: extractionSummary(inspection ?? { id: "", import_id: "", status: "completed", fields: [], preview_records: [], schema_candidates: [] }).documentCount ?? preparation?.estimates?.accepted ?? 0,
    character_count: characterCount,
    paragraph_count: texts.reduce((sum, text) => sum + Math.max(1, text.split(/\n\s*\n/).length), 0),
    byte_count: inspection?.size_bytes ?? characterCount,
    duplicate_documents: preparation?.estimates?.duplicates ?? 0,
    quarantined_documents: preparation?.estimates?.quarantined ?? 0,
    extraction_failures: extractionSummary(inspection ?? { id: "", import_id: "", status: "completed", fields: [], preview_records: [], schema_candidates: [] }).failed ?? 0,
    source_types: {},
  };
}

function localCorpusPackingPlan(profile: CorpusProfile, request: CorpusPackingRequest): CorpusPackingPlan {
  const estimatedCorpusTokens = Math.max(profile.document_count, Math.ceil(profile.character_count / 4));
  const trainAvailable = Math.max(1, Math.round(estimatedCorpusTokens * 0.9));
  const trainTokens = request.budget_mode === "tokens"
    ? Math.min(trainAvailable, request.target_tokens || trainAvailable)
    : Math.max(1, Math.round(trainAvailable * (request.corpus_passes || 1)));
  const validationTokens = Math.max(1, Math.round(estimatedCorpusTokens * 0.1));
  const trainBlocks = Math.max(1, Math.ceil(trainTokens / request.max_sequence_length));
  const validationBlocks = Math.max(1, Math.ceil(validationTokens / request.max_sequence_length));
  const paddingTokens = Math.max(0, trainBlocks * request.max_sequence_length - trainTokens);
  return {
    tokenizer_id: request.model,
    tokenizer_hash: "pending-tokenizer-aware-preview",
    max_sequence_length: request.max_sequence_length,
    separator: "eos",
    packing: request.packing,
    budget_mode: request.budget_mode,
    target_tokens: request.target_tokens,
    corpus_passes: request.corpus_passes,
    train_tokens: trainTokens,
    validation_tokens: validationTokens,
    train_blocks: trainBlocks,
    validation_blocks: validationBlocks,
    padding_tokens: paddingTokens,
    utilization: trainTokens / Math.max(1, trainBlocks * request.max_sequence_length),
    estimated_steps: Math.ceil(trainBlocks / Math.max(1, request.effective_batch_size || 1)),
    effective_batch_size: request.effective_batch_size || 1,
    warnings: ["Tokenizer-aware values will replace this local estimate when the packing route is available."],
  };
}

function unwrapCorpusPackingPlan(value?: CorpusPackingPlanResponse): CorpusPackingPlan | null {
  if (!value) return null;
  if ("train_tokens" in value) return value;
  return value.packing_plan ?? null;
}

function packingPreparationState(value?: CorpusPackingPlanResponse): { progress?: number | null; message?: string | null } | null {
  if (!value || "train_tokens" in value) return null;
  if (value.packing_plan || value.ready === true || ["completed", "succeeded"].includes(value.status)) return null;
  return { progress: value.progress_percent, message: value.message };
}

function friendlyRole(role: string): string {
  const labels: Record<string, string> = { user: "User", human: "User", assistant: "Assistant", system: "System", tool: "Tool result", function: "Tool result" };
  return labels[role.toLowerCase()] || humanize(role);
}

function asDisplayText(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) return value.slice(0, 4).map(asDisplayText).filter(Boolean).join(" · ");
  if (typeof value === "object") {
    return Object.entries(value as Record<string, unknown>)
      .filter(([key]) => !key.toLowerCase().endsWith("id") && !key.toLowerCase().includes("hash"))
      .slice(0, 5)
      .map(([key, nested]) => `${humanize(key)}: ${asDisplayText(nested)}`)
      .filter((item) => !item.endsWith(": "))
      .join(" · ");
  }
  return String(value);
}

function displayReference(value: unknown): string {
  if (typeof value === "string") return value.split(/[\\/]/).filter(Boolean).pop() || value;
  return asDisplayText(value) || "Referenced asset";
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : value === undefined || value === null ? [] : [value];
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function numberValue(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

async function createImportSession(draft: OwnDataDraft, scenario: TrainingScenarioDescriptor | undefined, files: File[]): Promise<DatasetImportSession> {
  if (!scenario) throw new Error("Choose a training scenario first.");
  if (draft.sourceMode === "upload" && !files.length) throw new Error("Choose at least one file or folder to upload.");
  if (draft.sourceMode === "upload" && files.some((file) => file.size === 0)) {
    const empty = files.find((file) => file.size === 0);
    throw new Error(`${empty?.webkitRelativePath || empty?.name || "An uploaded file"} is empty. Remove it or add content before inspection.`);
  }
  if (["desktop", "workstation"].includes(draft.sourceMode) && !draft.sourcePath.trim()) throw new Error("Choose or enter a workstation path.");
  if (draft.sourceMode === "huggingface" && (!draft.huggingFaceId.trim() || !draft.huggingFaceRevision.trim())) throw new Error("Enter a Hugging Face dataset ID and pinned revision.");
  if (draft.sourceMode === "upload" && draft.importId) {
    try {
      const existing = await api.datasetImport(draft.importId);
      if (existing.source_kind === "upload" && existing.scenario_revision_id === scenario.revision_id && !["expired", "published"].includes(existing.status)) return existing;
    } catch {
      // The saved import may have been cleaned up. Creating a new durable
      // session is safe because every chunk is checksummed independently.
    }
  }
  return api.createDatasetImport({
    source_kind: draft.sourceMode === "desktop" ? "desktop" : draft.sourceMode === "workstation" ? "reference" : draft.sourceMode,
    scenario_revision_id: scenario.revision_id,
    source_uri: draft.sourceMode === "desktop" || draft.sourceMode === "workstation" ? draft.sourcePath.trim() : draft.sourceMode === "huggingface" ? draft.huggingFaceId.trim() : undefined,
    config: draft.huggingFaceConfig.trim() || undefined,
    split: draft.huggingFaceSplit.trim() || undefined,
    revision: draft.huggingFaceRevision.trim() || undefined,
    example_id: draft.sourceMode === "example" ? draft.exampleId || undefined : undefined,
    expected_size_bytes: draft.sourceMode === "upload" ? files.reduce((sum, file) => sum + file.size, 0) : undefined,
    capacity_override_reason: draft.capacityOverrideReason.trim() || undefined,
  });
}

function unwrapInspection(value: Awaited<ReturnType<typeof api.inspectDatasetImport>>): DatasetSourceInspection | null {
  if (value.inspection) return value.inspection;
  if (typeof value.id === "string" && Array.isArray(value.fields)) return value as DatasetSourceInspection;
  return null;
}

function unwrapDataset(value: Awaited<ReturnType<typeof api.registerInspectedDataset>>) {
  if (value.dataset) return value.dataset;
  if (typeof value.id === "string" && Array.isArray(value.sources)) return value;
  return null;
}

function isGuidedScenario(scenario: TrainingScenarioDescriptor): boolean {
  if (!scenario.available || scenario.verified === false) return false;
  if (scenario.modality === "audio" && !["asr", "automatic_speech_recognition", "speech_recognition", "transcription"].includes(String(scenario.task_type || "asr").toLowerCase())) return false;
  return true;
}

function findScenario(scenarios: TrainingScenarioDescriptor[], id: string): TrainingScenarioDescriptor | undefined { return scenarios.find((scenario) => scenario.revision_id === id || scenario.id === id); }
function scenarioFields(scenario: TrainingScenarioDescriptor, required: boolean): TrainingScenarioField[] { const values = required ? scenario.required_fields : scenario.optional_fields ?? []; return values.map((field) => typeof field === "string" ? { name: field, label: humanize(field), required } : { ...field, required }); }
function preferredTrainerMode(scenario: TrainingScenarioDescriptor): string { return String(scenario.trainer_modes?.[0] ?? scenario.compatible_trainers?.find((item) => item.compatible)?.trainer_mode ?? (scenario.canonical_shape === "preference" ? "dpo" : scenario.canonical_shape === "vlm" ? "vlm" : scenario.canonical_shape === "audio" ? "audio" : scenario.canonical_shape === "tool" ? "agentic" : scenario.canonical_shape === "rlvr" ? "raft" : "sft")); }
function defaultDatasetName(scenario: TrainingScenarioDescriptor, draft: OwnDataDraft): string { const source = draft.sourcePath.split("/").filter(Boolean).pop()?.replace(/\.[^.]+$/, "") || draft.huggingFaceId.split("/").pop() || scenario.label; return `${source} · ${scenarioKindLabel(scenario)}`; }

function buildSuggestedMapping(candidate: SchemaCandidate, scenario: TrainingScenarioDescriptor | undefined, inspection: DatasetSourceInspection): FieldMappingPlan {
  const mappings: Record<string, FieldMappingExpression> = {};
  const suggested = candidate.suggested_mapping ?? {};
  const mediaRoot = inspection.preview_records.find((record) => typeof record._media_root === "string")?._media_root as string | undefined;
  if (scenario) {
    for (const field of [...scenarioFields(scenario, true), ...scenarioFields(scenario, false)]) {
      const suggestion = suggested[field.name];
      if (typeof suggestion === "string") mappings[field.name] = { kind: "direct", source: suggestion };
      else if (suggestion && typeof suggestion === "object" && "kind" in suggestion) mappings[field.name] = suggestion as FieldMappingExpression;
      else {
        const aliases = [field.name, ...(field.aliases ?? []), ...(scenario.accepted_aliases?.[field.name] ?? [])].map((value) => value.toLowerCase());
        const match = inspection.fields.find((source) => aliases.includes(source.name.toLowerCase()));
        if (match) mappings[field.name] = field.name.includes("image") || field.name.includes("audio") ? { kind: "media_root", source: match.name, root: mediaRoot || "" } : field.name === "messages" ? { kind: "conversation", source: match.name, role_field: "role", content_field: "content", role_map: { human: "user", assistant: "assistant", system: "system" } } : { kind: "direct", source: match.name };
      }
    }
  }
  return { version: 2, scenario_revision_id: candidate.scenario_revision_id, mappings, confirmed: false };
}

function buildLocalMappingPreview(inspection: DatasetSourceInspection, plan: FieldMappingPlan): MappingPreview {
  const items = inspection.preview_records.slice(0, 5).map((source, ordinal) => {
    const canonical: Record<string, unknown> = {};
    const issues: Array<{ field?: string; message: string; severity?: string }> = [];
    for (const [target, expression] of Object.entries(plan.mappings)) {
      const value = applyMappingExpression(source, expression);
      canonical[target] = value;
      if (value === undefined || value === null || value === "") issues.push({ field: target, message: `${humanize(target)} is empty`, severity: "error" });
    }
    return { ordinal, source, canonical, issues };
  });
  const valid = items.filter((item) => !item.issues.length).length;
  return { items, total_sampled: items.length, valid_count: valid, invalid_count: items.length - valid, ready: items.length > 0 && valid > 0 };
}

function applyMappingExpression(source: Record<string, unknown>, expression: FieldMappingExpression): unknown {
  if (expression.kind === "direct") return source[expression.source];
  if (expression.kind === "constant") return expression.value;
  if (expression.kind === "concat") return expression.sources.map((field) => source[field]).filter((value) => value !== undefined && value !== null).join(expression.separator ?? "\n");
  if (expression.kind === "nested_path") return getPath(source[expression.source], expression.path);
  if (expression.kind === "media_root") { const value = source[expression.source]; return typeof value === "string" && expression.root && expression.root !== "." ? `${expression.root.replace(/\/$/, "")}/${value.replace(/^\//, "")}` : value; }
  if (expression.kind === "conversation") { const value = source[expression.source]; if (!Array.isArray(value)) return value; return value.map((message) => { if (!message || typeof message !== "object") return message; const record = message as Record<string, unknown>; const role = String(record[expression.role_field || "role"] ?? ""); return { role: expression.role_map?.[role] ?? role, content: record[expression.content_field || "content"] }; }); }
}

function getPath(value: unknown, path: string): unknown { let current = value; for (const segment of path.split(".").filter(Boolean)) { if (current === null || current === undefined) return undefined; current = (current as Record<string, unknown>)[segment]; } return current; }
function defaultExpression(kind: FieldMappingExpression["kind"], source: string, mediaRoot?: string): FieldMappingExpression { if (kind === "constant") return { kind, value: "" }; if (kind === "concat") return { kind, sources: source ? [source] : [], separator: "\n" }; if (kind === "nested_path") return { kind, source, path: "" }; if (kind === "conversation") return { kind, source, role_field: "role", content_field: "content", role_map: { human: "user", assistant: "assistant", system: "system" } }; if (kind === "media_root") return { kind, source, root: mediaRoot || "" }; return { kind: "direct", source }; }
function mappingExpressionReady(expression?: FieldMappingExpression): boolean { if (!expression) return false; if (expression.kind === "media_root") return Boolean(expression.source && expression.root); if (expression.kind === "direct" || expression.kind === "conversation" || expression.kind === "nested_path") return Boolean(expression.source); if (expression.kind === "concat") return expression.sources.length > 0; return expression.value !== undefined && expression.value !== ""; }

function resolvedRecipe(draft: OwnDataDraft, scenario?: TrainingScenarioDescriptor): DatasetRecipe {
  if (draft.advancedRecipe && draft.rawRecipe) { try { return parseRecipeText(draft.rawRecipe); } catch { /* surfaced separately */ } }
  const settings = draft.preparation;
  const corpus = isCorpusScenario(scenario);
  const steps: DatasetRecipe["steps"] = [];
  if (draft.repairRevisionId) steps.push({ kind: "repair_overlay", revision_id: draft.repairRevisionId });
  if (draft.mappingPlan) steps.push({ kind: "map", mapping_version: 2, scenario_revision_id: scenario?.revision_id, fields: draft.mappingPlan.mappings });
  if (corpus) {
    steps.push({
      kind: "document_clean",
      strip_boilerplate: settings.stripBoilerplate,
      preserve_headings: settings.preserveHeadings,
      preserve_code_blocks: settings.preserveHeadings,
    });
    steps.push({
      kind: "document_filter",
      quarantine_extraction_failures: settings.quarantineExtractionFailures,
      require_visible_text: true,
    });
  }
  if (settings.normalizeWhitespace) steps.push({ kind: "normalize", trim: true, collapse_whitespace: !corpus });
  if (settings.validate) steps.push({ kind: "validate", on_error: settings.quarantineInvalid || settings.quarantineExtractionFailures ? "quarantine" : "reject" });
  if (settings.exactDedup) steps.push({ kind: "dedup", method: "exact", scope: "canonical_record" });
  if (corpus && settings.fuzzyDedup) steps.push({ kind: "dedup", method: "fuzzy", fields: ["text"], threshold: 0.92 });
  const mediaGroupField = scenario?.modality === "image" ? "image" : scenario?.modality === "audio" ? "audio" : undefined;
  const splitGroup = corpus && settings.preserveDocumentBoundaries ? "source_ref" : settings.groupMedia ? mediaGroupField : undefined;
  steps.push({ kind: "split", method: splitGroup ? "grouped" : "random", group_field: splitGroup, ratios: { train: settings.trainRatio / 100, validation: settings.validationRatio / 100, test: settings.testRatio / 100 }, seed: 42 });
  if (settings.contamination) steps.push({ kind: "contamination", action: "report", splits: ["train", "validation", "test", "canary"] });
  return { name: corpus ? "guided-corpus-adaptation" : "guided-own-data-v1", kind: "ordered", schema: scenario?.canonical_shape, seed: 42, steps };
}

function parseRecipeText(value: string): DatasetRecipe {
  const parsed = parseYaml(value) as unknown;
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("Recipe must be a YAML or JSON object.");
  const record = parsed as Record<string, unknown>;
  if (!Array.isArray(record.steps)) throw new Error("Recipe must include an ordered steps array.");
  return record as DatasetRecipe;
}

function sourceReady(draft: OwnDataDraft, files: File[], examples: Array<{ id: string }>): boolean { if (draft.sourceMode === "upload") return files.length > 0 && files.every((file) => file.size > 0); if (draft.sourceMode === "desktop" || draft.sourceMode === "workstation") return Boolean(draft.sourcePath.trim()); if (draft.sourceMode === "huggingface") return Boolean(draft.huggingFaceId.trim() && draft.huggingFaceRevision.trim()); if (draft.sourceMode === "example") return Boolean(draft.exampleId || examples[0]?.id); return false; }
function stepComplete(step: StudioStep, draft: OwnDataDraft, inspection?: DatasetSourceInspection | null, versionId?: string): boolean { if (step === "goal") return Boolean(draft.scenarioRevisionId); if (step === "source") return Boolean(draft.importId); if (step === "format") return Boolean(inspection?.status === "completed" && draft.candidateConfirmed); if (step === "map") return draft.mappingConfirmed; if (step === "prepare") return Boolean(draft.mappingConfirmed); if (step === "version") return Boolean(versionId); return Boolean(draft.proofRunId); }
function canEnterStep(step: StudioStep, draft: OwnDataDraft, inspection?: DatasetSourceInspection | null, versionId?: string): boolean { const index = STEPS.findIndex((item) => item.id === step); if (index === 0) return true; const previous = STEPS[index - 1].id; return stepComplete(previous, draft, inspection, versionId); }
function cycleTrainer(mode: string): boolean { return ["raft", "grpo", "reasoning", "agentic", "vlm", "audio"].includes(mode); }

function scenarioIcon(scenario: TrainingScenarioDescriptor): LucideIcon { if (scenario.modality === "audio") return AudioLines; if (scenario.modality === "image" || scenario.canonical_shape === "vlm") return Image; if (scenario.canonical_shape === "preference") return ShieldCheck; if (scenario.canonical_shape === "tool") return Wrench; if (scenario.canonical_shape === "chat") return MessageSquareText; if (scenario.task_type?.includes("code")) return Code2; return FileJson; }
function humanize(value: string): string { return value.replaceAll("_", " ").replace(/\b\w/g, (match) => match.toUpperCase()); }
function formatInteger(value?: number | null): string { return typeof value === "number" ? new Intl.NumberFormat().format(value) : "—"; }
function formatBytes(value?: number | null): string { if (typeof value !== "number") return "—"; const units = ["B", "KB", "MB", "GB", "TB"]; let amount = value; let unit = 0; while (amount >= 1024 && unit < units.length - 1) { amount /= 1024; unit += 1; } return `${amount >= 10 || unit === 0 ? amount.toFixed(0) : amount.toFixed(1)} ${units[unit]}`; }
function sameStrings(left: string[], right: string[]): boolean { return left.length === right.length && left.every((value, index) => value === right[index]); }
function percent(value?: number | null): string { if (typeof value !== "number") return "—"; return `${Math.round((value <= 1 ? value * 100 : value) * 10) / 10}%`; }
function shortHash(value: string): string { return value.length > 16 ? `${value.slice(0, 12)}…` : value; }
async function sha256Hex(content: ArrayBuffer): Promise<string> {
  if (!globalThis.crypto?.subtle) throw new Error("This browser cannot checksum uploads. Use a supported browser or reference a workstation path instead.");
  const digest = await globalThis.crypto.subtle.digest("SHA-256", content);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}
function workspaceScrollContainer(): HTMLElement | null {
  const main = document.getElementById("main");
  if (!main) return null;
  return Array.from(main.children).find((element): element is HTMLElement => element instanceof HTMLElement && ["auto", "scroll"].includes(window.getComputedStyle(element).overflowY)) ?? null;
}
