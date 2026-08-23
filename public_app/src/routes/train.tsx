import { createFileRoute, Link } from "@tanstack/react-router";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  AlertTriangle,
  AudioLines,
  BookOpen,
  Brain,
  CheckCircle2,
  CircleDashed,
  Code2,
  Copy,
  Database,
  Eye,
  GitCompareArrows,
  Loader2,
  Package,
  Play,
  Settings2,
  ShieldCheck,
  Sparkles,
  Wrench,
  XCircle,
  type LucideIcon,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { api } from "@/lib/api";
import { Topbar } from "@/components/shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardEyebrow, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { SearchPicker } from "@/components/ui/search-picker";
import { RadioCard, RadioCardGroup } from "@/components/ui/radio-card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  useBackendInfo,
  useModelCatalog,
  useTrainingDatasets,
  useTrainingLaunch,
  useTrainingModels,
  useTrainingPreflight,
  useTrainingVerifiers,
  useWorkspaceInfo,
} from "@/lib/hooks";
import type {
  BackendInfo,
  DatasetBinding,
  DatasetJob,
  DatasetVersion,
  ModelCatalogEntry,
  TrainingMode,
  TrainingDatasetArtifact,
  TrainingTemplate,
  VerifierProfile,
  RewardIntegrityForkContext,
} from "@/lib/api";
import { cn } from "@/lib/utils";
import {
  EMPTY_REWARD_AUDIT_BINDING,
  RewardAuditBindingEditor,
  type RewardAuditBindingValue,
} from "@/components/research/reward-audit-binding";

export const Route = createFileRoute("/train")({
  component: TrainConfiguratorRoute,
  validateSearch: (search): {
    template?: string;
    model?: string;
    mode?: string;
    datasetVersion?: string;
    datasetSplit?: string;
    parentRun?: string;
    fork_reward_audit?: string;
    goal?: string;
  } => ({
    template: typeof search.template === "string" ? search.template : undefined,
    model: typeof search.model === "string" ? search.model : undefined,
    mode: typeof search.mode === "string" ? search.mode : undefined,
    datasetVersion:
      typeof search.datasetVersion === "string" ? search.datasetVersion : undefined,
    datasetSplit:
      typeof search.datasetSplit === "string" ? search.datasetSplit : undefined,
    parentRun: typeof search.parentRun === "string" ? search.parentRun : undefined,
    fork_reward_audit:
      typeof search.fork_reward_audit === "string" ? search.fork_reward_audit : undefined,
    goal: typeof search.goal === "string" ? search.goal : undefined,
  }),
});

type GoalKey = "code" | "reasoning" | "tool-use" | "vision" | "audio" | "preferences" | "task-models";
type Accelerator = "auto" | "mlx";
type TrainingSource = {
  key: string;
  description: string;
  size_hint: string;
  domain: string;
  huggingface_id?: string;
};

const TRAINING_MODES: TrainingMode[] = [
  "sft",
  "raft",
  "dpo",
  "orpo",
  "rm",
  "grpo",
  "vlm",
  "audio",
  "reasoning",
  "agentic",
  "classify",
  "embed",
  "rerank",
];

const DEFAULT_RAFT_PROMPTS = "data/rlvr/humaneval_prompts.jsonl";

const GOALS: Array<{
  key: GoalKey;
  label: string;
  description: string;
  icon: LucideIcon;
  modes: TrainingMode[];
}> = [
  {
    key: "task-models",
    label: "Task models",
    description: "Classification, embeddings, and reranking with compact specialist models.",
    icon: Package,
    modes: ["classify", "embed", "rerank"],
  },
  {
    key: "code",
    label: "Code",
    description: "Instruction tuning, compiler-filtered RAFT, or verifier RL for programming tasks.",
    icon: Code2,
    modes: ["sft", "raft", "grpo"],
  },
  {
    key: "reasoning",
    label: "Reasoning",
    description: "Math and multi-step tasks using SFT, reasoning adapters, or GRPO.",
    icon: Brain,
    modes: ["sft", "reasoning", "grpo"],
  },
  {
    key: "tool-use",
    label: "Tool use",
    description: "Function calling, structured outputs, schema checks, and agent traces.",
    icon: Wrench,
    modes: ["sft", "agentic", "grpo"],
  },
  {
    key: "vision",
    label: "Vision",
    description: "Vision-language fine-tuning for VQA, documents, charts, and extraction.",
    icon: Eye,
    modes: ["vlm"],
  },
  {
    key: "audio",
    label: "Audio",
    description: "Speech and audio training paths for Whisper-class models.",
    icon: AudioLines,
    modes: ["audio"],
  },
  {
    key: "preferences",
    label: "Preferences",
    description: "Chosen/rejected pair training: DPO, ORPO, reward models, then GRPO.",
    icon: ShieldCheck,
    modes: ["dpo", "orpo", "rm", "grpo"],
  },
];

const METHOD_COPY: Record<TrainingMode, { label: string; description: string; caveat?: string }> = {
  sft: {
    label: "SFT",
    description: "Learn from labeled examples. Best first step for almost every project.",
  },
  cpt: {
    label: "Continued pretraining",
    description: "Adapt a causal language model to a reviewed document corpus.",
    caveat: "Use the guided own-data workflow to preserve extraction and packing provenance.",
  },
  raft: {
    label: "RAFT",
    description: "Generate, verify, keep, train. Simple verifier-grounded improvement.",
  },
  dpo: {
    label: "DPO",
    description: "Preference tuning from prompt/chosen/rejected pairs.",
  },
  orpo: {
    label: "ORPO",
    description: "Reference-free preference tuning from the same pair data as DPO.",
  },
  rm: {
    label: "Reward model",
    description: "Train a scorer on chosen/rejected pairs for later RL or ranking.",
  },
  grpo: {
    label: "GRPO",
    description: "Verifier-grounded RL with group-relative advantages.",
    caveat: "Needs a verifier and more memory than SFT.",
  },
  vlm: {
    label: "Vision-language",
    description: "Domain training for image plus text datasets.",
    caveat: "May require prototype capability approval depending on model family.",
  },
  audio: {
    label: "Audio",
    description: "Speech/audio training with task-specific processors.",
    caveat: "Requires audio dependencies and task selection.",
  },
  reasoning: {
    label: "Reasoning",
    description: "Reasoning-specific RAFT-style training and math datasets.",
  },
  agentic: {
    label: "Agentic",
    description: "Tool-use and function-calling traces with structured verification.",
  },
  classify: {
    label: "Classification",
    description: "Train a text, image, or audio classification head from reviewed labels.",
  },
  embed: {
    label: "Embeddings",
    description: "Train a bi-encoder with in-batch multiple-negative ranking loss.",
  },
  rerank: {
    label: "Reranker",
    description: "Train a cross-encoder from relevance scores or reviewed ordering.",
  },
};

const METHOD_GUIDANCE: Record<TrainingMode, { headline: string; detail: string }> = {
  sft: {
    headline: "Best default when you have examples.",
    detail: "Halo Forge will train on the selected dataset and write the run under the workstation run folder.",
  },
  cpt: {
    headline: "Adapt a model to a document corpus.",
    detail: "Use Train on your data to extract documents, publish an immutable corpus, preview packing, and set an explicit token or pass budget.",
  },
  raft: {
    headline: "Good when answers can be checked.",
    detail: "The run generates candidate answers, verifies them, keeps the passing samples, then trains.",
  },
  dpo: {
    headline: "Use when you have chosen/rejected pairs.",
    detail: "DPO tunes the model toward preferred answers while comparing against a reference policy.",
  },
  orpo: {
    headline: "Preference tuning without a reference model.",
    detail: "ORPO uses the same pair format as DPO and is lighter to set up for many local runs.",
  },
  rm: {
    headline: "Build a scorer before RL or ranking.",
    detail: "The reward model learns to score chosen answers above rejected answers.",
  },
  grpo: {
    headline: "Advanced verifier-grounded RL.",
    detail: "Use this after a small SFT or RAFT run; it needs a verifier and usually more memory.",
  },
  vlm: {
    headline: "Vision-language training path.",
    detail: "Use image-plus-text datasets and keep the prototype gate enabled only when the backend asks for it.",
  },
  audio: {
    headline: "Audio training path.",
    detail: "Choose the audio task first; dependencies and model family determine whether launch is available.",
  },
  reasoning: {
    headline: "Reasoning-specific training path.",
    detail: "Use math or multi-step datasets when you want chain quality rather than general instruction following.",
  },
  agentic: {
    headline: "Tool-use and function-calling path.",
    detail: "Use structured traces and schema checks when the model needs to call tools reliably.",
  },
  classify: {
    headline: "Predict one or more reviewed classes.",
    detail: "Halo Forge verifies the label map, processor, model head, and a fixed-input round trip.",
  },
  embed: {
    headline: "Train retrieval representations.",
    detail: "Anchor-positive pairs use a verified bi-encoder objective and preserve retrieval-corpus identity.",
  },
  rerank: {
    headline: "Improve candidate ordering.",
    detail: "Query-document relevance trains a verified cross-encoder scoring head.",
  },
};

const DEFAULTS: Record<TrainingMode, Partial<ConfigState>> = {
  sft: { dataset: "codealpaca", epochs: 1, batchSize: 2, learningRate: "2e-4", maxSamples: 200 },
  cpt: { dataset: "", epochs: 1, batchSize: 1, learningRate: "2e-5" },
  raft: { dataset: DEFAULT_RAFT_PROMPTS, cycles: 1, samplesPerPrompt: 4, verifier: "execution" },
  dpo: { dataset: "ultrafeedback", epochs: 1, batchSize: 1, learningRate: "5e-6", beta: "0.1", lossType: "sigmoid" },
  orpo: { dataset: "ultrafeedback", epochs: 1, batchSize: 1, learningRate: "8e-6", beta: "0.1" },
  rm: { dataset: "ultrafeedback", epochs: 1, batchSize: 4, learningRate: "1e-5" },
  grpo: { dataset: "gsm8k", epochs: 1, batchSize: 1, learningRate: "1e-6", beta: "0.04", verifier: "json_schema", numGenerations: 4 },
  vlm: { dataset: "textvqa", cycles: 1, samplesPerPrompt: 2, maxSamples: 24 },
  audio: { dataset: "librispeech", cycles: 1, samplesPerPrompt: 2, task: "asr" },
  reasoning: { dataset: "gsm8k", cycles: 1, maxSamples: 64, learningRate: "1e-5" },
  agentic: { dataset: "xlam_sft", cycles: 1, maxSamples: 64, learningRate: "5e-5" },
  classify: { dataset: "", epochs: 1, batchSize: 4, learningRate: "2e-5", maxSamples: 200 },
  embed: { dataset: "", epochs: 1, batchSize: 8, learningRate: "2e-5", maxSamples: 200 },
  rerank: { dataset: "", epochs: 1, batchSize: 8, learningRate: "2e-5", maxSamples: 200 },
};

const RAFT_PROMPT_ALIASES: Record<string, string> = {
  humaneval: "data/rlvr/humaneval_prompts.jsonl",
  mbpp: "data/rlvr/mbpp_train_prompts.jsonl",
  codeforces_cpp: "data/samples/codeforces_cpp_500_prompts.jsonl",
};

const RAFT_PROMPT_SOURCES: TrainingSource[] = [
  {
    key: DEFAULT_RAFT_PROMPTS,
    description: "HumanEval coding prompts for verifier-filtered RAFT.",
    size_hint: "164 prompts",
    domain: "code prompts",
  },
  {
    key: "data/rlvr/mbpp_train_prompts.jsonl",
    description: "MBPP training prompts for Python coding RAFT.",
    size_hint: "374 prompts",
    domain: "code prompts",
  },
  {
    key: "data/samples/codeforces_cpp_500_prompts.jsonl",
    description: "Codeforces C++ prompt-only sample set.",
    size_hint: "500 prompts",
    domain: "code prompts",
  },
];

const PREFERENCE_SOURCES: TrainingSource[] = [
  { key: "ultrafeedback", description: "General preference pairs for chat quality.", size_hint: "large", domain: "preference" },
  { key: "ultrafeedback-binarized", description: "Binarized chosen/rejected UltraFeedback pairs.", size_hint: "large", domain: "preference" },
  { key: "orca_dpo", description: "ORCA-style preference pairs.", size_hint: "medium", domain: "preference" },
  { key: "hh_rlhf", description: "Helpful/harmless preference pairs.", size_hint: "medium", domain: "preference" },
  { key: "py_dpo", description: "Python/code preference pairs.", size_hint: "small", domain: "preference" },
];

const MODALITY_SOURCES: Record<TrainingMode, TrainingSource[]> = {
  sft: [],
  cpt: [],
  raft: RAFT_PROMPT_SOURCES,
  dpo: PREFERENCE_SOURCES,
  orpo: PREFERENCE_SOURCES,
  rm: PREFERENCE_SOURCES,
  grpo: [
    { key: "gsm8k", description: "Math prompts for answer-verifier GRPO.", size_hint: "small", domain: "reasoning" },
    ...RAFT_PROMPT_SOURCES,
  ],
  vlm: [
    { key: "textvqa", description: "Image question answering.", size_hint: "medium", domain: "vision" },
    { key: "docvqa", description: "Document visual question answering.", size_hint: "medium", domain: "vision" },
    { key: "vqa-rad", description: "Medical VQA smoke dataset.", size_hint: "small", domain: "vision" },
  ],
  audio: [
    { key: "librispeech", description: "Speech-to-text ASR data.", size_hint: "large", domain: "audio" },
    { key: "librispeech-clean", description: "Clean audiobook ASR subset.", size_hint: "medium", domain: "audio" },
  ],
  reasoning: [
    { key: "gsm8k", description: "Math word problems.", size_hint: "small", domain: "reasoning" },
    { key: "gsm8k_sft", description: "SFT-formatted GSM8K examples.", size_hint: "small", domain: "reasoning" },
  ],
  agentic: [
    { key: "xlam_sft", description: "Tool-use and function-calling examples.", size_hint: "small", domain: "agentic" },
    { key: "glaive_sft", description: "Function-calling instruction examples.", size_hint: "medium", domain: "agentic" },
  ],
  classify: [],
  embed: [],
  rerank: [],
};

interface ConfigState {
  goal: GoalKey;
  modality: TrainingMode;
  model: string;
  dataset: string;
  customDatasetFile: string;
  datasetVersionId: string;
  datasetSplit: string;
  datasetBindings: DatasetBinding[];
  parentRunId: string;
  forkRewardAuditId: string;
  forkRewardDecisionId: string;
  forkCheckpointHash: string;
  forkCheckpointPath: string;
  forkCheckpointOccurrenceId: string;
  forkCheckpointSnapshotPath: string;
  forkBoundaryUnit: string;
  forkBoundaryValue: number;
  forkResumeMode: string;
  accelerator: Accelerator;
  verifier: string;
  verifierProfileRevisionId: string;
  rewardAudit: RewardAuditBindingValue;
  task: string;
  epochs: number;
  batchSize: number;
  learningRate: string;
  seed: number;
  cycles: number;
  samplesPerPrompt: number;
  maxSamples: number;
  beta: string;
  lossType: string;
  referenceFree: boolean;
  numGenerations: number;
  rewardThreshold: string;
  allowPrototypeTrain: boolean;
  templateId: string | null;
}

function defaultConfig(): ConfigState {
  return {
    goal: "code",
    modality: "sft",
    model: "",
    dataset: "codealpaca",
    customDatasetFile: "",
    datasetVersionId: "",
    datasetSplit: "train",
    datasetBindings: [],
    parentRunId: "",
    forkRewardAuditId: "",
    forkRewardDecisionId: "",
    forkCheckpointHash: "",
    forkCheckpointPath: "",
    forkCheckpointOccurrenceId: "",
    forkCheckpointSnapshotPath: "",
    forkBoundaryUnit: "",
    forkBoundaryValue: 0,
    forkResumeMode: "",
    accelerator: "auto",
    verifier: "execution",
    verifierProfileRevisionId: "",
    rewardAudit: EMPTY_REWARD_AUDIT_BINDING,
    task: "asr",
    epochs: 1,
    batchSize: 2,
    learningRate: "2e-4",
    seed: 42,
    cycles: 1,
    samplesPerPrompt: 4,
    maxSamples: 200,
    beta: "0.1",
    lossType: "sigmoid",
    referenceFree: false,
    numGenerations: 4,
    rewardThreshold: "0.0",
    allowPrototypeTrain: false,
    templateId: null,
  };
}

function TrainConfiguratorRoute() {
  const backend = useBackendInfo();
  const managedRuntimeCapabilities = useQuery({
    queryKey: ["managed-runtime-capabilities"],
    queryFn: () => api.managedRuntimeCapabilities(),
    retry: false,
    refetchInterval: 15_000,
  });
  const workspace = useWorkspaceInfo();
  const datasets = useTrainingDatasets();
  const verifiers = useTrainingVerifiers();
  const preflight = useTrainingPreflight();
  const launch = useTrainingLaunch();
  const {
    template: templateId,
    model: modelId,
    mode,
    datasetVersion,
    datasetSplit,
    parentRun,
    fork_reward_audit: forkRewardAuditId,
    goal,
  } = Route.useSearch();

  const [config, setConfig] = useState<ConfigState>(defaultConfig);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [launchedRun, setLaunchedRun] = useState<Record<string, unknown> | null>(null);
  const retriedArtifactJobs = useRef(new Set<string>());
  const verifierProfiles = useQuery({
    queryKey: ["verifier-profiles", "train", config.modality],
    queryFn: () => api.listVerifierProfiles({ qualification: "pass", limit: 200 }),
    enabled: ["raft", "grpo", "reasoning", "agentic", "vlm", "audio"].includes(config.modality),
    retry: false,
  });
  const models = useTrainingModels({ mode: config.modality });
  const managedVersions = useQuery({
    queryKey: ["training", "dataset-versions", config.modality],
    queryFn: () => api.trainingDatasetVersions(config.modality),
  });
  const managedArtifacts = useQuery({
    queryKey: ["dataset-versions", config.datasetVersionId, "training-artifacts"],
    queryFn: () => api.listTrainingArtifacts(config.datasetVersionId),
    enabled: Boolean(config.datasetVersionId),
    refetchInterval: (query) =>
      query.state.data?.items.some((artifact) => ["queued", "rendering", "running"].includes(artifact.status))
        ? 2_000
        : false,
    retry: false,
  });
  const mlxModels = useModelCatalog({ mode: config.modality, backend: "mlx" });
  const mlxReadiness = backend.data?.mlx_readiness;
  const mlxReady = mlxReadiness?.executable === true;
  const managedFamily = backend.data?.name?.startsWith("rocm")
    ? "rocm"
    : backend.data?.name === "cuda" ? "cuda" : null;
  const managedRuntime = managedRuntimeCapabilities.data?.items.find(
    (item) => item.accelerator_family === managedFamily,
  );
  const trainingPaths = useQuery({
    queryKey: ["training-paths", managedFamily],
    queryFn: () => api.trainingPaths(managedFamily as "rocm" | "cuda"),
    enabled: Boolean(managedFamily),
    retry: false,
    refetchInterval: 15_000,
  });
  const selectedTrainingPath = trainingPaths.data?.paths.find(
    (item) => item.trainer_mode === config.modality && item.model_id === config.model,
  );
  const certifySelectedPath = useMutation({
    mutationFn: () => api.certifyTrainingPath(
      selectedTrainingPath!.path_revision_id,
      selectedTrainingPath!.runtime_revision_id!,
    ),
    onSuccess: () => void trainingPaths.refetch(),
  });
  const modelSuggestions = useMemo(
    () => (mlxReady ? (mlxModels.data?.items ?? []) : (models.data?.items ?? [])),
    [mlxReady, mlxModels.data?.items, models.data?.items],
  );
  const allCatalogModels = useMemo(
    () => [...(models.data?.items ?? []), ...(mlxModels.data?.items ?? [])],
    [models.data?.items, mlxModels.data?.items],
  );
  const selectedModel = useMemo(
    () => allCatalogModels.find((item) => item.id === config.model) ?? null,
    [allCatalogModels, config.model],
  );
  const payload = useMemo(
    () => buildLaunchPayload(config, workspace.data?.default_run_root),
    [config, workspace.data?.default_run_root],
  );
  const pendingArtifactJobId =
    preflight.isSuccess && preflight.data.status === "preparing_dataset"
      ? preflight.data.job_id ?? null
      : null;
  const artifactPreparationJob = useQuery({
    queryKey: ["dataset-jobs", pendingArtifactJobId],
    queryFn: () => api.datasetJob(pendingArtifactJobId!),
    enabled: Boolean(pendingArtifactJobId),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return !status || ["queued", "running"].includes(status) ? 1_000 : false;
    },
    retry: false,
  });
  const currentPreflightStatus = preflightStatus(
    preflight,
    config,
    artifactPreparationJob.data,
  );

  const templateQuery = useQuery({
    queryKey: ["training-template", templateId],
    queryFn: () => api.trainingTemplate(templateId!),
    enabled: Boolean(templateId),
  });
  const parentConfigQuery = useQuery({
    queryKey: ["runs", parentRun, "launch-config"],
    queryFn: () => api.runLaunchConfig(parentRun!),
    enabled: Boolean(parentRun),
    retry: false,
  });
  const rewardForkQuery = useQuery({
    queryKey: ["reward-integrity-audit", forkRewardAuditId, "fork-context"],
    queryFn: () => api.rewardIntegrityForkContext(forkRewardAuditId!),
    enabled: Boolean(forkRewardAuditId),
    retry: false,
  });

  useEffect(() => {
    if (!goal) return;
    if (goal === "apple-silicon") {
      setConfig((prev) => ({ ...prev, goal: "code", modality: "sft", accelerator: "mlx" }));
      return;
    }
    if (isGoalKey(goal)) {
      const suggestedMode = GOALS.find((item) => item.key === goal)?.modes[0] ?? "sft";
      setConfig((prev) => ({ ...prev, goal, modality: suggestedMode }));
    }
  }, [goal]);

  useEffect(() => {
    if (!modelId) return;
    setConfig((prev) => ({
      ...prev,
      model: modelId,
      modality: isTrainingMode(mode) ? mode : prev.modality,
      accelerator: isMlxModel(modelId) ? "mlx" : prev.accelerator,
    }));
  }, [modelId, mode]);

  useEffect(() => {
    if (!isTrainingMode(mode)) return;
    setConfig((prev) => ({ ...prev, modality: mode }));
  }, [mode]);

  useEffect(() => {
    if (!datasetVersion) return;
    const selectedSplit = datasetSplit || "train";
    setConfig((prev) => ({
      ...prev,
      datasetVersionId: datasetVersion,
      datasetSplit: selectedSplit,
      datasetBindings: upsertBinding(prev.datasetBindings, {
        role: "train",
        dataset_version_id: datasetVersion,
        split: selectedSplit,
      }),
    }));
  }, [datasetVersion, datasetSplit]);

  useEffect(() => {
    if (!parentRun || !parentConfigQuery.data || config.parentRunId === parentRun) return;
    setConfig((prev) =>
      applyResolvedLaunchConfig(
        prev,
        parentRun,
        parentConfigQuery.data.resolved_config,
        parentConfigQuery.data.datasets,
      ),
    );
  }, [parentRun, parentConfigQuery.data, config.parentRunId]);

  useEffect(() => {
    if (!forkRewardAuditId || !rewardForkQuery.data || config.forkRewardAuditId === forkRewardAuditId) return;
    setConfig((prev) => applyRewardAuditForkContext(prev, rewardForkQuery.data));
  }, [forkRewardAuditId, rewardForkQuery.data, config.forkRewardAuditId]);

  useEffect(() => {
    if (!templateQuery.data || config.templateId === templateQuery.data.id) return;
    setConfig((prev) => applyTemplate(prev, templateQuery.data));
  }, [templateQuery.data, config.templateId]);

  useEffect(() => {
    if (modelId || config.model || !modelSuggestions.length || config.templateId) return;
    const first = modelSuggestions[0].id;
    setConfig((prev) => ({
      ...prev,
      model: first,
      accelerator: mlxReady && isMlxModel(first) ? "mlx" : prev.accelerator,
      batchSize: mlxReady && isMlxModel(first) ? 1 : prev.batchSize,
    }));
  }, [modelSuggestions, config.model, config.templateId, modelId, mlxReady]);

  useEffect(() => {
    if (!canLaunch(config)) return;
    const t = window.setTimeout(() => {
      preflight.mutate(buildLaunchPayload(config, workspace.data?.default_run_root));
    }, 400);
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    config.modality,
    config.model,
    config.dataset,
    config.customDatasetFile,
    config.datasetVersionId,
    config.datasetSplit,
    config.datasetBindings,
    config.parentRunId,
    config.forkRewardAuditId,
    config.forkRewardDecisionId,
    config.forkCheckpointHash,
    config.forkCheckpointPath,
    config.forkCheckpointOccurrenceId,
    config.forkCheckpointSnapshotPath,
    config.forkBoundaryUnit,
    config.forkBoundaryValue,
    config.forkResumeMode,
    config.accelerator,
    config.verifier,
    config.verifierProfileRevisionId,
    config.rewardAudit,
    config.task,
    config.epochs,
    config.batchSize,
    config.learningRate,
    config.seed,
    config.cycles,
    config.samplesPerPrompt,
    config.maxSamples,
    config.beta,
    config.lossType,
    config.referenceFree,
    config.numGenerations,
    config.rewardThreshold,
    config.allowPrototypeTrain,
    workspace.data?.default_run_root,
  ]);

  useEffect(() => {
    const job = artifactPreparationJob.data;
    if (!job || !["completed", "succeeded"].includes(job.status)) return;
    if (retriedArtifactJobs.current.has(job.id)) return;
    retriedArtifactJobs.current.add(job.id);
    void managedArtifacts.refetch();
    preflight.mutate(payload);
    // The mutation/query objects are stable React Query handles; key this retry
    // strictly to the persisted job transition and current launch payload.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [artifactPreparationJob.data?.id, artifactPreparationJob.data?.status, payload]);

  return (
    <>
      <Topbar
        eyebrow="Workspace"
        title="Train"
        subtitle="Choose a managed dataset version, goal, and method; Halo Forge generates a conservative launch you can inspect."
        actions={
          <>
            <Button asChild variant="primary" size="sm">
              <Link to="/datasets/new" search={{ example: undefined }}>
                <Database />
                Train on your data
              </Link>
            </Button>
            <Button asChild variant="secondary" size="sm">
              <Link to="/datasets/new" search={{ example: "1" }}>
                <Sparkles />
                Try a working example
              </Link>
            </Button>
            <Button asChild variant="ghost" size="sm">
              <Link to="/train/templates">
                <Sparkles />
                Templates
              </Link>
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                setConfig(defaultConfig());
                setLaunchedRun(null);
              }}
              type="button"
            >
              Reset
            </Button>
          </>
        }
        statusBar={
          <>
            <ReadoutItem label="GOAL" value={goalLabel(config.goal)} />
            <ReadoutSep />
            <ReadoutItem label="METHOD" value={config.modality.toUpperCase()} />
            <ReadoutSep />
            <ReadoutItem label="BACKEND" value={backend.data?.name ?? "-"} />
            <ReadoutSep />
            <ReadoutItem label="MLX" value={mlxReady ? "READY" : mlxReadiness?.status?.toUpperCase() ?? "-"} />
          </>
        }
      />

      <div className="px-5 py-5 space-y-4">
        <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_340px] gap-4">
          <div className="space-y-4">
            {managedFamily ? (
              <div className={`flex flex-wrap items-center justify-between gap-3 border-l-2 px-4 py-3 ${selectedTrainingPath?.state === "path_verified" ? "border-success bg-success-bg" : "border-warning bg-warning-bg"}`}>
                <div>
                  <p className="text-xs font-medium text-fg">{!managedRuntime?.available ? "Training runtime needs preparation" : selectedTrainingPath?.state === "path_verified" ? "This training path is verified" : "This training path needs verification"}</p>
                  <p className="mt-1 text-[11px] leading-5 text-fg-muted">{!managedRuntime?.available ? "Hardware detection alone is not treated as training readiness. Prepare and qualify the managed runtime before launching." : selectedTrainingPath?.summary ?? "Choose a model with a real certification profile. Generic tensor checks do not unlock guided training."}</p>
                </div>
                {!managedRuntime?.available ? <Button asChild size="sm" variant="primary"><Link to="/setup">Prepare {managedFamily === "rocm" ? "AMD" : "NVIDIA"} training</Link></Button> : selectedTrainingPath?.state === "unavailable" ? <Button size="sm" variant="secondary" disabled><ShieldCheck />Not available yet</Button> : selectedTrainingPath && selectedTrainingPath.state !== "path_verified" ? <Button size="sm" variant="primary" onClick={() => certifySelectedPath.mutate()} disabled={certifySelectedPath.isPending || selectedTrainingPath.state === "verification_in_progress"}>{certifySelectedPath.isPending || selectedTrainingPath.state === "verification_in_progress" ? <Loader2 className="animate-spin" /> : <ShieldCheck />}{selectedTrainingPath.state === "verification_in_progress" ? "Verifying in Activity" : "Verify this training path"}</Button> : null}
              </div>
            ) : null}
            {parentRun ? (
              <ForkContext
                runId={parentRun}
                loading={parentConfigQuery.isLoading}
                error={parentConfigQuery.isError ? (parentConfigQuery.error as Error).message : null}
              />
            ) : null}
            {forkRewardAuditId ? (
              <RewardAuditForkContext
                auditId={forkRewardAuditId}
                context={rewardForkQuery.data}
                loading={rewardForkQuery.isLoading}
                error={rewardForkQuery.isError ? (rewardForkQuery.error as Error).message : null}
              />
            ) : null}
            <GoalSection
              config={config}
              onChange={(goal) =>
                setConfig((prev) => {
                  const nextMode = GOALS.find((item) => item.key === goal)?.modes[0] ?? "sft";
                  return withModeDefaults({ ...prev, goal }, nextMode);
                })
              }
            />
            <MethodSection
              config={config}
              onChange={(mode) => setConfig((prev) => withModeDefaults(prev, mode))}
            />
            <LaunchInputs
              config={config}
              setConfig={setConfig}
              datasets={datasets.data?.items ?? []}
              verifiers={verifiers.data?.items ?? []}
              qualifiedVerifierProfiles={verifierProfiles.data?.items ?? []}
              modelSuggestions={modelSuggestions}
              selectedModel={selectedModel}
              mlxReady={mlxReady}
              managedVersions={managedVersions.data?.items ?? []}
              managedArtifacts={managedArtifacts.data?.items ?? []}
              managedArtifactsLoading={managedArtifacts.isLoading}
            />
            <RewardAuditBindingEditor
              trainerMode={config.modality}
              backendFamily={config.accelerator === "mlx" || isMlxModel(config.model) ? "mlx" : "hf"}
              value={config.rewardAudit}
              onChange={(rewardAudit) => setConfig((prev) => ({ ...prev, rewardAudit }))}
              totalBudget={cycleMode(config.modality) ? config.cycles : config.maxSamples}
              budgetUnit={cycleMode(config.modality) ? "cycle" : "step"}
            />
            <AdvancedOptions
              config={config}
              setConfig={setConfig}
              open={advancedOpen}
              onOpenChange={setAdvancedOpen}
            />
            {launchedRun ? <LaunchSuccess data={launchedRun} payload={payload} /> : null}
          </div>

          <div className="xl:sticky xl:top-4 self-start space-y-3">
            <PreflightPanel
              preflightStatus={currentPreflightStatus}
              checks={buildPreflightChecks(
                config,
                preflight,
                backend.data?.name,
                mlxReadiness,
                artifactPreparationJob.data,
              )}
            />
            <LaunchPanel
              config={config}
              payload={payload}
              selectedModel={selectedModel}
              preflight={preflight}
              artifactPreparationJob={artifactPreparationJob.data}
              launch={launch}
              onLaunched={(data) => setLaunchedRun(data)}
            />
          </div>
        </div>
      </div>
    </>
  );
}

function GoalSection({ config, onChange }: { config: ConfigState; onChange: (goal: GoalKey) => void }) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardEyebrow>STEP 01</CardEyebrow>
          <CardTitle>Goal</CardTitle>
        </div>
      </CardHeader>
      <CardContent>
        <RadioCardGroup
          value={config.goal}
          onValueChange={(value) => onChange(value as GoalKey)}
          className="grid gap-2 md:grid-cols-2 xl:grid-cols-3"
        >
          {GOALS.map((goal) => (
            <RadioCard
              key={goal.key}
              value={goal.key}
              title={goal.label}
              description={
                <span className="flex gap-2">
                  <goal.icon className="mt-0.5 h-3.5 w-3.5 shrink-0 text-accent" />
                  <span>{goal.description}</span>
                </span>
              }
            />
          ))}
        </RadioCardGroup>
      </CardContent>
    </Card>
  );
}

function MethodSection({ config, onChange }: { config: ConfigState; onChange: (mode: TrainingMode) => void }) {
  const activeGoal = GOALS.find((goal) => goal.key === config.goal) ?? GOALS[0];
  const modes = activeGoal.modes;
  const guidance = METHOD_GUIDANCE[config.modality];
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardEyebrow>STEP 02</CardEyebrow>
          <CardTitle>Training method</CardTitle>
        </div>
        <Button asChild variant="ghost" size="sm">
          <a href="https://halo-forge.io/docs/training-pipeline/methods/" target="_blank" rel="noreferrer">
            <BookOpen />
            Method guide
          </a>
        </Button>
      </CardHeader>
      <CardContent>
        <RadioCardGroup
          value={config.modality}
          onValueChange={(value) => onChange(value as TrainingMode)}
          className="grid gap-2 md:grid-cols-2"
        >
          {modes.map((mode) => (
            <RadioCard
              key={mode}
              value={mode}
              title={METHOD_COPY[mode].label}
              badge={<Badge size="sm" tone={mode === "sft" ? "success" : "neutral"}>{mode}</Badge>}
              description={
                <span>
                  {METHOD_COPY[mode].description}
                  {METHOD_COPY[mode].caveat ? (
                    <span className="mt-1 block text-warning">{METHOD_COPY[mode].caveat}</span>
                  ) : null}
                </span>
              }
            />
          ))}
        </RadioCardGroup>
        <div className="mt-3 rounded-md border border-border-subtle bg-bg-subtle px-3 py-2">
          <div className="text-[12px] font-medium text-fg">{guidance.headline}</div>
          <div className="mt-1 text-[12px] text-fg-muted">{guidance.detail}</div>
        </div>
      </CardContent>
    </Card>
  );
}

function LaunchInputs({
  config,
  setConfig,
  datasets,
  verifiers,
  qualifiedVerifierProfiles,
  modelSuggestions,
  selectedModel,
  mlxReady,
  managedVersions,
  managedArtifacts,
  managedArtifactsLoading,
}: {
  config: ConfigState;
  setConfig: (updater: (c: ConfigState) => ConfigState) => void;
  datasets: TrainingSource[];
  verifiers: Array<{ key: string; label: string; toolchain: string }>;
  qualifiedVerifierProfiles: VerifierProfile[];
  modelSuggestions: ModelCatalogEntry[];
  selectedModel: ModelCatalogEntry | null;
  mlxReady: boolean;
  managedVersions: DatasetVersion[];
  managedArtifacts: TrainingDatasetArtifact[];
  managedArtifactsLoading: boolean;
}) {
  const sources = sourcesForMode(config.modality, datasets);
  const isCustom = config.dataset === "__custom__";
  const isManaged = Boolean(config.datasetVersionId);
  const requiresVerifier = config.modality === "raft" || config.modality === "grpo";
  const supportsVerifierProfile = supportsVerifierBinding(config.modality);
  const trainBinding = config.datasetBindings.find((binding) => binding.role === "train");
  const compatibleVerifierProfiles = qualifiedVerifierProfiles.filter((profile) => {
    const revision = profile.latest_revision;
    return revision
      && revision.qualification_state === "pass"
      && revision.overridden !== true
      && revision.runtime_compatible !== false
      && isVerifierModalityCompatible(revision.modality, config.modality);
  });

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardEyebrow>STEP 03</CardEyebrow>
          <CardTitle>Generated launch</CardTitle>
        </div>
        {selectedModel ? <Badge tone="info" size="sm">{selectedModel.memory_tier}</Badge> : null}
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-3 lg:grid-cols-2">
          <FormField label="Base model">
            <Input
              value={config.model}
              onChange={(event) => setConfig((prev) => ({ ...prev, model: event.target.value }))}
              placeholder="Qwen/Qwen2.5-1.5B-Instruct"
            />
          </FormField>
          <FormField label="accelerator">
            <Select
              value={config.accelerator}
              onValueChange={(value) => setConfig((prev) => ({ ...prev, accelerator: value as Accelerator }))}
            >
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="auto">Auto</SelectItem>
                <SelectItem value="mlx">MLX</SelectItem>
              </SelectContent>
            </Select>
          </FormField>
        </div>

        {modelSuggestions.length ? (
          <div className="flex flex-wrap gap-1.5">
            {modelSuggestions.slice(0, 4).map((model) => (
              <button
                key={model.id}
                type="button"
                onClick={() =>
                  setConfig((prev) => ({
                    ...prev,
                    model: model.id,
                    accelerator: mlxReady && isMlxModel(model.id) ? "mlx" : prev.accelerator,
                  }))
                }
                className={cn(
                  "rounded-sm border px-2 py-1 font-mono text-[11px]",
                  config.model === model.id
                    ? "border-accent bg-accent/10 text-accent"
                    : "border-border-subtle text-fg-subtle hover:text-fg",
                )}
              >
                {model.id}
              </button>
            ))}
          </div>
        ) : null}

        <div className="space-y-3 rounded-md border border-border-subtle bg-bg-subtle/25 p-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <div className="text-[11px] font-medium text-fg">Managed dataset version</div>
              <div className="mt-0.5 text-[10px] text-fg-muted">Select the immutable version prepared in Data. Its supplied validation split is preserved exactly.</div>
            </div>
            {isManaged ? <Badge tone="success" dot size="sm">ready for preflight</Badge> : <Badge tone="warning" size="sm">choose data</Badge>}
          </div>
          <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_180px]">
            <FormField label="Dataset version">
              <SearchPicker
                value={config.datasetVersionId}
                onChange={(versionId) => {
                  const version = managedVersions.find((item) => item.id === versionId);
                  const splits = Object.keys(version?.split_counts ?? {});
                  const split = splits.includes("train") ? "train" : splits[0] || "train";
                  setConfig((prev) => ({
                    ...prev,
                    datasetVersionId: versionId,
                    datasetSplit: split,
                    datasetBindings: versionId
                      ? upsertBinding(prev.datasetBindings.filter((binding) => binding.role !== "train"), { role: "train", dataset_version_id: versionId, split })
                      : prev.datasetBindings.filter((binding) => binding.role !== "train"),
                  }));
                }}
                options={managedVersions.filter((version) => ["ready", "completed"].includes(version.status)).map((version, index) => ({
                  value: version.id,
                  label: version.label || `Dataset version ${index + 1}`,
                  description: `${version.row_count?.toLocaleString() || "—"} rows · ${Object.keys(version.split_counts || {}).join(", ") || "train"}`,
                  status: version.status,
                  keywords: `${version.content_hash || ""} ${version.recipe_hash || ""}`,
                }))}
                placeholder="Choose a compatible prepared version"
                emptyLabel="No prepared version is available for this method"
              />
            </FormField>
            <FormField label="Training split">
              <Select
                value={config.datasetSplit}
                disabled={!config.datasetVersionId}
                onValueChange={(split) => setConfig((prev) => ({ ...prev, datasetSplit: split, datasetBindings: upsertBinding(prev.datasetBindings, { role: "train", dataset_version_id: prev.datasetVersionId, split }) }))}
              >
                <SelectTrigger aria-label="Training split"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {(Object.keys(managedVersions.find((version) => version.id === config.datasetVersionId)?.split_counts ?? {}).filter((split) => !["test", "canary"].includes(split)).length
                    ? Object.keys(managedVersions.find((version) => version.id === config.datasetVersionId)?.split_counts ?? {}).filter((split) => !["test", "canary"].includes(split))
                    : ["train"]).map((split) => <SelectItem key={split} value={split}>{split}</SelectItem>)}
                </SelectContent>
              </Select>
            </FormField>
          </div>
          {!isManaged ? (
            <div className="flex flex-wrap items-center justify-between gap-3 border-l-2 border-accent bg-accent-bg/45 px-3 py-2.5">
              <div><div className="text-[11px] font-medium text-fg">Need to prepare your source?</div><div className="mt-0.5 text-[10px] text-fg-muted">The guided flow inspects, maps, splits, and validates it before training.</div></div>
              <div className="flex flex-wrap gap-2"><Button variant="primary" size="sm" asChild><Link to="/datasets/new" search={{ example: undefined }}><Database />Train on your data</Link></Button><Button variant="secondary" size="sm" asChild><Link to="/datasets/new" search={{ example: "1" }}><Sparkles />Try a working example</Link></Button></div>
            </div>
          ) : null}
          {isManaged ? (
            <ManagedArtifactStatus
              artifacts={managedArtifacts}
              mode={config.modality}
              loading={managedArtifactsLoading}
            />
          ) : null}
          <details className="border-t border-border-subtle pt-2">
            <summary className="cursor-pointer text-[9.5px] uppercase tracking-wider text-fg-disabled hover:text-fg">Advanced · roles, built-ins, and manual paths</summary>
            <div className="mt-3 space-y-4">
              <div>
                <div className="mb-1 text-[10px] font-medium text-fg">Managed dataset bindings</div>
                <div className="mb-2 text-[10px] text-fg-muted">Assign additional immutable versions by role. Test and canary are stored for evaluation only and never reach the trainer.</div>
                <DatasetBindingEditor
                  versions={managedVersions}
                  bindings={config.datasetBindings}
                  onChange={(bindings) => setConfig((prev) => {
                    const nextTrain = bindings.find((binding) => binding.role === "train");
                    return { ...prev, datasetBindings: bindings, datasetVersionId: nextTrain?.dataset_version_id || "", datasetSplit: nextTrain?.split || "train" };
                  })}
                />
              </div>
              {!isManaged ? (
                <div className="grid gap-3 border-t border-border-subtle pt-3 lg:grid-cols-2">
                  <FormField label={sourceLabel(config.modality)}>
                    <Select value={config.dataset} onValueChange={(value) => setConfig((prev) => ({ ...prev, dataset: value }))}>
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>{sources.map((source) => <SelectItem key={source.key} value={source.key}>{source.key}</SelectItem>)}<SelectItem value="__custom__">Custom local file</SelectItem></SelectContent>
                    </Select>
                  </FormField>
                  {isCustom ? <FormField label="Local JSONL path"><Input value={config.customDatasetFile} onChange={(event) => setConfig((prev) => ({ ...prev, customDatasetFile: event.target.value }))} placeholder="/path/to/training.jsonl" /></FormField> : <FormField label="Dataset note"><div className="min-h-9 rounded-md border border-border bg-bg-subtle px-3 py-2 text-[12px] text-fg-muted">{sources.find((source) => source.key === config.dataset)?.description ?? "Registered dataset or source."}</div></FormField>}
                </div>
              ) : null}
              {trainBinding ? <Button type="button" size="sm" variant="ghost" onClick={() => setConfig((prev) => ({ ...prev, datasetBindings: [], datasetVersionId: "", datasetSplit: "train" }))}>Clear managed bindings</Button> : null}
            </div>
          </details>
        </div>

        {supportsVerifierProfile ? (
          <div className="grid gap-3 lg:grid-cols-2">
            <FormField label={requiresVerifier ? "Qualified verifier profile" : "Qualified verifier profile · optional"}>
              <SearchPicker
                value={config.verifierProfileRevisionId}
                onChange={(value) => setConfig((prev) => ({ ...prev, verifierProfileRevisionId: value, verifier: value ? "" : prev.verifier }))}
                options={compatibleVerifierProfiles.flatMap((profile) => profile.latest_revision ? [{ value: profile.latest_revision.id, label: profile.name, description: `${profile.latest_revision.family.replace("_", " ")} · ${profile.latest_revision.modality} · ${profile.latest_revision.alias || profile.latest_revision.qualification_state || "qualified"}`, status: profile.latest_revision.qualification_state, keywords: `${profile.latest_revision.content_hash || ""} ${profile.description || ""}` }] : [])}
                placeholder="Choose a compatible qualified verifier"
                emptyLabel="No pass-qualified verifier profile is available"
              />
              {requiresVerifier ? (
                <details className="mt-2">
                  <summary className="cursor-pointer text-[9.5px] uppercase tracking-wider text-fg-disabled hover:text-fg">
                    Advanced · legacy raw verifier
                  </summary>
                  <div className="mt-2">
                    <div className="mb-1 text-[9.5px] font-medium text-fg-subtle">Verifier toolchain</div>
                    <Select
                      value={config.verifier || "execution"}
                      onValueChange={(value) => setConfig((prev) => ({ ...prev, verifier: value, verifierProfileRevisionId: "" }))}
                    >
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {verifiers.map((verifier) => <SelectItem key={verifier.key} value={verifier.key}>{verifier.key}</SelectItem>)}
                        <SelectItem value="json_schema">json_schema</SelectItem>
                        <SelectItem value="llm_judge">llm_judge</SelectItem>
                      </SelectContent>
                    </Select>
                    <p className="mt-1 text-[9.5px] leading-4 text-warning">
                      Raw verifiers remain runnable as legacy unqualified inputs and cannot provide exact reliability replay.
                    </p>
                  </div>
                </details>
              ) : null}
            </FormField>
            <FormField label="Verifier note">
              <div className="min-h-9 rounded-md border border-border bg-bg-subtle px-3 py-2 text-[12px] text-fg-muted">
                {config.verifierProfileRevisionId
                  ? "Exact profile revision, qualification scope, and runtime identity are captured in replay."
                  : !requiresVerifier
                    ? "Optional reliability binding for this trainer. Guided selection only shows compatible pass-qualified revisions."
                    : config.modality === "grpo"
                    ? "GRPO uses this legacy unqualified verifier as the reward function."
                    : "RAFT uses this legacy unqualified verifier to filter generations."}
              </div>
            </FormField>
          </div>
        ) : null}

        {config.modality === "audio" ? (
          <FormField label="Audio task">
            <Select value="asr" onValueChange={() => setConfig((prev) => ({ ...prev, task: "asr" }))}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent><SelectItem value="asr">Speech recognition (ASR)</SelectItem></SelectContent></Select>
            <p className="mt-1 text-[10px] leading-4 text-fg-subtle">Guided audio training currently supports Whisper-style transcription. Classification and text-to-speech remain hidden until they have verified trainer contracts.</p>
          </FormField>
        ) : null}
      </CardContent>
    </Card>
  );
}

const DATASET_ROLES = ["train", "validation", "test", "canary"] as const;

function DatasetBindingEditor({
  versions,
  bindings,
  onChange,
}: {
  versions: DatasetVersion[];
  bindings: DatasetBinding[];
  onChange: (bindings: DatasetBinding[]) => void;
}) {
  function setVersion(role: string, versionId: string) {
    const withoutRole = bindings.filter((binding) => binding.role !== role);
    if (!versionId) return onChange(withoutRole);
    const version = versions.find((item) => item.id === versionId);
    const splits = Object.keys(version?.split_counts ?? {});
    const split = splits.includes(role)
      ? role
      : role === "validation" && splits.includes("val")
        ? "val"
        : splits.includes("train")
          ? "train"
          : splits[0] || "train";
    onChange([...withoutRole, { role, dataset_version_id: versionId, split }]);
  }

  function setSplit(role: string, split: string) {
    onChange(bindings.map((binding) => binding.role === role ? { ...binding, split } : binding));
  }

  return (
    <div className="divide-y divide-border-subtle border-y border-border-subtle">
      {DATASET_ROLES.map((role) => {
        const binding = bindings.find((item) => item.role === role);
        const selectedVersion = versions.find((version) => version.id === binding?.dataset_version_id);
        const splits = Array.from(new Set([
          ...(binding?.split ? [binding.split] : []),
          ...Object.keys(selectedVersion?.split_counts ?? {}),
        ]));
        return (
          <div key={role} className="grid gap-2 py-2 md:grid-cols-[90px_minmax(220px,1fr)_150px] md:items-center">
            <div>
              <span className="text-[10px] font-medium uppercase tracking-[0.12em] text-fg-subtle">{role}</span>
              {role === "train" ? <span className="ml-1 text-danger">*</span> : null}
            </div>
            <select
              value={binding?.dataset_version_id || ""}
              onChange={(event) => setVersion(role, event.target.value)}
              className="h-8 min-w-0 rounded-md border border-border bg-bg px-2 font-mono text-[10.5px] text-fg"
            >
              <option value="">{role === "train" ? "Choose compatible version" : "Not bound"}</option>
              {binding && !selectedVersion ? <option value={binding.dataset_version_id}>{binding.dataset_version_id}</option> : null}
              {versions.map((version) => <option key={version.id} value={version.id}>{version.label || version.id} · {formatRowCount(version.row_count)}</option>)}
            </select>
            <select
              value={binding?.split || ""}
              onChange={(event) => setSplit(role, event.target.value)}
              disabled={!binding}
              className="h-8 rounded-md border border-border bg-bg px-2 font-mono text-[10.5px] text-fg disabled:opacity-40"
            >
              {!binding ? <option value="">No split</option> : null}
              {(splits.length ? splits : [binding?.split || "train"]).map((split) => <option key={split} value={split}>{split}{selectedVersion?.split_counts?.[split] !== undefined ? ` · ${selectedVersion.split_counts[split]}` : ""}</option>)}
            </select>
          </div>
        );
      })}
    </div>
  );
}

function ManagedArtifactStatus({ artifacts, mode, loading }: { artifacts: TrainingDatasetArtifact[]; mode: TrainingMode; loading: boolean }) {
  const artifact = artifacts.find((item) => item.trainer_mode === mode && item.status === "ready")
    ?? artifacts.find((item) => item.trainer_mode === mode);
  if (loading) return <div className="flex items-center gap-2 border-t border-border-subtle pt-2 text-[10px] text-fg-muted"><Loader2 className="h-3 w-3 animate-spin text-accent" />Checking trainer artifact readiness…</div>;
  if (!artifact) return <div className="flex items-center gap-2 border-t border-border-subtle pt-2 text-[10px] text-fg-muted"><Package className="h-3 w-3 text-fg-disabled" />A content-addressed {mode} artifact will be prepared atomically during preflight.</div>;
  const progress = Math.max(0, Math.min(100, artifact.progress_percent ?? (artifact.status === "ready" ? 100 : 0)));
  return <div className="border-t border-border-subtle pt-2"><div className="flex items-center justify-between gap-3"><div className="min-w-0"><div className="flex items-center gap-2"><Package className="h-3 w-3 text-accent" /><span className="text-[10px] font-medium text-fg">{artifact.adapter_id}@{artifact.adapter_version}</span><Badge tone={artifact.status === "ready" ? "success" : artifact.status === "failed" ? "danger" : "accent"} dot size="sm">{artifact.status}</Badge></div><div className="mt-1 truncate font-mono text-[9.5px] text-fg-disabled">{artifact.artifact_hash || artifact.stage || artifact.id}</div></div><span className="font-mono text-[10px] text-fg-muted">{progress.toFixed(0)}%</span></div><div className="mt-2 h-1 overflow-hidden rounded-full bg-bg-subtle"><div className={cn("h-full transition-all", artifact.status === "failed" ? "bg-danger" : "bg-accent")} style={{ width: `${progress}%` }} /></div></div>;
}

function ForkContext({ runId, loading, error }: { runId: string; loading: boolean; error: string | null }) {
  return (
    <Card className="border-accent/35 bg-accent/5">
      <CardContent className="flex flex-wrap items-center justify-between gap-3 px-4 py-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-[12px] font-medium text-fg">
            {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin text-accent" /> : <Copy className="h-3.5 w-3.5 text-accent" />}
            Clone in Train
          </div>
          <div className="mt-1 truncate font-mono text-[10.5px] text-fg-muted">parent_run_id = {runId}</div>
          <div className="mt-0.5 text-[10.5px] text-fg-subtle">The resolved launch config and dataset bindings are prefilled; the backend records the exact diff on launch.</div>
          {error ? <div className="mt-1 text-[10.5px] text-danger">Could not load resolved config: {error}</div> : null}
        </div>
        <Button variant="ghost" size="sm" asChild><Link to="/runs/$runId" params={{ runId }}>Open parent</Link></Button>
      </CardContent>
    </Card>
  );
}

function RewardAuditForkContext({
  auditId,
  context,
  loading,
  error,
}: {
  auditId: string;
  context?: RewardIntegrityForkContext;
  loading: boolean;
  error: string | null;
}) {
  const checkpoint = context?.checkpoint;
  const boundary = checkpoint
    ? `${checkpoint.boundary_unit || "boundary"} ${checkpoint.boundary_value ?? "final"}`
    : "audited boundary";
  return (
    <Card className="border-accent/35 bg-accent/5">
      <CardContent className="px-4 py-3">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2 text-[12px] font-medium text-fg">
              {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin text-accent" /> : <GitCompareArrows className="h-3.5 w-3.5 text-accent" />}
              Fork from reviewed checkpoint
            </div>
            <div className="mt-1 text-[10.5px] text-fg-subtle">
              {context ? `${boundary} · ${context.resume_mode === "resume_boundary" ? "resume exact trainer state" : "initialize from the published checkpoint"}` : "Resolving immutable audit and checkpoint lineage…"}
            </div>
          </div>
          {context?.parent_run_id ? <Button variant="ghost" size="sm" asChild><Link to="/runs/$runId" params={{ runId: context.parent_run_id }}>Open parent</Link></Button> : null}
        </div>
        {checkpoint ? <dl className="mt-3 grid gap-2 border-t border-border-subtle pt-3 sm:grid-cols-2"><div><dt className="text-[9px] uppercase tracking-wider text-fg-disabled">Checkpoint hash</dt><dd className="mt-0.5 truncate font-mono text-[9.5px] text-fg-muted" title={checkpoint.content_hash}>{checkpoint.content_hash}</dd></div><div><dt className="text-[9px] uppercase tracking-wider text-fg-disabled">Artifact</dt><dd className="mt-0.5 truncate font-mono text-[9.5px] text-fg-muted">{checkpoint.occurrence_id || "sealed checkpoint path"}</dd></div></dl> : null}
        {context && !context.launch_ready ? <div role="alert" className="mt-3 border-l-2 border-danger bg-danger/5 px-3 py-2 text-[10px] text-danger">This fork cannot launch until the checkpoint is available: {context.blockers.join(", ")}.</div> : null}
        {error ? <div role="alert" className="mt-3 text-[10.5px] text-danger">Could not restore the reviewed fork: {error}</div> : null}
        <div className="mt-2 font-mono text-[8.5px] text-fg-disabled">audit {auditId}</div>
      </CardContent>
    </Card>
  );
}

function AdvancedOptions({
  config,
  setConfig,
  open,
  onOpenChange,
}: {
  config: ConfigState;
  setConfig: (updater: (c: ConfigState) => ConfigState) => void;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  return (
    <Collapsible open={open} onOpenChange={onOpenChange}>
      <Card>
        <CollapsibleTrigger asChild>
          <CardHeader className="cursor-pointer hover:bg-surface-hover/30">
            <div className="flex items-center gap-2">
              <CardEyebrow>OPTIONAL</CardEyebrow>
              <CardTitle>Advanced settings</CardTitle>
            </div>
            <Settings2 className="h-4 w-4 text-fg-subtle" />
          </CardHeader>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <CardContent className="space-y-3">
            <div className="rounded-md border border-border-subtle bg-bg-subtle px-3 py-2 text-[12px] text-fg-muted">
              These are here for repeat runs and method experiments. The generated defaults are the safest first launch.
            </div>
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
            <NumberField label="Epochs" value={config.epochs} onChange={(value) => setConfig((prev) => ({ ...prev, epochs: value }))} />
            <NumberField label="Batch size" value={config.batchSize} onChange={(value) => setConfig((prev) => ({ ...prev, batchSize: value }))} />
            <FormField label="Learning rate">
              <Input value={config.learningRate} onChange={(event) => setConfig((prev) => ({ ...prev, learningRate: event.target.value }))} />
            </FormField>
            <NumberField label="Seed" value={config.seed} onChange={(value) => setConfig((prev) => ({ ...prev, seed: value }))} />
            <NumberField label="Max samples / limit" value={config.maxSamples} onChange={(value) => setConfig((prev) => ({ ...prev, maxSamples: value }))} />
            {cycleMode(config.modality) ? (
              <>
                <NumberField label="Cycles" value={config.cycles} onChange={(value) => setConfig((prev) => ({ ...prev, cycles: value }))} />
                <NumberField label="Samples per prompt" value={config.samplesPerPrompt} onChange={(value) => setConfig((prev) => ({ ...prev, samplesPerPrompt: value }))} />
              </>
            ) : null}
            {preferenceMode(config.modality) || config.modality === "grpo" ? (
              <FormField label="Beta">
                <Input value={config.beta} onChange={(event) => setConfig((prev) => ({ ...prev, beta: event.target.value }))} />
              </FormField>
            ) : null}
            {config.modality === "dpo" ? (
              <FormField label="DPO loss">
                <Select value={config.lossType} onValueChange={(value) => setConfig((prev) => ({ ...prev, lossType: value }))}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="sigmoid">sigmoid</SelectItem>
                    <SelectItem value="ipo">ipo</SelectItem>
                    <SelectItem value="hinge">hinge</SelectItem>
                    <SelectItem value="kto_pair">kto_pair</SelectItem>
                  </SelectContent>
                </Select>
              </FormField>
            ) : null}
            {config.modality === "grpo" ? (
              <>
                <NumberField label="Generations" value={config.numGenerations} onChange={(value) => setConfig((prev) => ({ ...prev, numGenerations: value }))} />
                <FormField label="Reward threshold">
                  <Input value={config.rewardThreshold} onChange={(event) => setConfig((prev) => ({ ...prev, rewardThreshold: event.target.value }))} />
                </FormField>
              </>
            ) : null}
            {(config.modality === "dpo" || config.modality === "grpo") ? (
              <label className="flex items-center gap-2 rounded-md border border-border-subtle bg-bg-subtle px-3 py-2 text-[12px] text-fg-muted">
                <input
                  type="checkbox"
                  checked={config.referenceFree}
                  onChange={(event) => setConfig((prev) => ({ ...prev, referenceFree: event.target.checked }))}
                />
                Reference-free
              </label>
            ) : null}
            {["vlm", "audio", "reasoning", "agentic"].includes(config.modality) ? (
              <label className="flex items-center gap-2 rounded-md border border-warning/30 bg-warning-bg px-3 py-2 text-[12px] text-warning">
                <input
                  type="checkbox"
                  checked={config.allowPrototypeTrain}
                  onChange={(event) => setConfig((prev) => ({ ...prev, allowPrototypeTrain: event.target.checked }))}
                />
                Enable prototype method when the backend requires it
              </label>
            ) : null}
            </div>
          </CardContent>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  );
}

function LaunchPanel({
  config,
  payload,
  selectedModel,
  preflight,
  artifactPreparationJob,
  launch,
  onLaunched,
}: {
  config: ConfigState;
  payload: Record<string, unknown>;
  selectedModel: ModelCatalogEntry | null;
  preflight: ReturnType<typeof useTrainingPreflight>;
  artifactPreparationJob?: DatasetJob;
  launch: ReturnType<typeof useTrainingLaunch>;
  onLaunched: (data: Record<string, unknown>) => void;
}) {
  const [reservedRunId, setReservedRunId] = useState<string | null>(null);
  const payloadIdentity = JSON.stringify(payload);
  useEffect(() => setReservedRunId(null), [payloadIdentity]);
  const preparingArtifact =
    preflight.isSuccess &&
    (preflight.data.status === "preparing_dataset" || preflight.data.ready === false);
  const artifactFailed = Boolean(
    preparingArtifact &&
      artifactPreparationJob &&
      ["failed", "cancelled"].includes(artifactPreparationJob.status),
  );
  const readyToLaunch =
    preflight.isSuccess && preflight.data.ok && !preparingArtifact && !artifactFailed;
  const disabled = !canLaunch(config) || launch.isPending || !readyToLaunch;
  const launchCopy = launchHint(config.modality);
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardEyebrow>LAUNCH</CardEyebrow>
          <CardTitle>Launch summary</CardTitle>
        </div>
        <Badge tone={readyToLaunch ? "success" : artifactFailed ? "danger" : "neutral"} size="sm">
          {config.modality}
        </Badge>
      </CardHeader>
      <CardContent className="space-y-3">
        <SummaryRows
          rows={[
            ["method", config.modality],
            ["model", String(payload.model ?? "-")],
            [
              sourceLabel(config.modality).toLowerCase(),
              String(payload.dataset_version_id ?? payload.dataset ?? payload.prompts ?? "-"),
            ],
            ["output root", String(payload.output_root ?? payload.output_dir ?? "-")],
            ["memory", selectedModel?.estimated_memory_gb ? `~${selectedModel.estimated_memory_gb}GB` : selectedModel?.memory_tier ?? "-"],
          ]}
        />
        <div className="rounded-sm border border-border-subtle bg-bg-subtle px-2 py-1.5">
          <div className="text-[11px] font-medium text-fg">{launchCopy.headline}</div>
          <div className="mt-0.5 text-[11px] text-fg-muted">{launchCopy.detail}</div>
        </div>
        {selectedModel?.known_caveats?.length ? (
          <div className="rounded-sm border border-warning/30 bg-warning-bg px-2 py-1.5 text-[11px] text-warning">
            {selectedModel.known_caveats[0]}
          </div>
        ) : null}
        {launch.isError ? (
          <div className="rounded-sm border border-danger/30 bg-danger-bg px-2 py-1.5 text-[11px] text-danger">
            {(launch.error as Error).message}
          </div>
        ) : null}
        <Button
          variant="primary"
          size="lg"
          className="w-full"
          disabled={disabled}
          onClick={() => {
            const launchPayload = reservedRunId
              ? { ...payload, run_id: reservedRunId }
              : payload;
            launch.mutate(launchPayload, {
              onSuccess: (data) => {
                if (data.status === "preparing_dataset") {
                  const runId = typeof data.run_id === "string" ? data.run_id : null;
                  setReservedRunId(runId);
                  preflight.mutate(runId ? { ...payload, run_id: runId } : payload);
                  return;
                }
                setReservedRunId(null);
                onLaunched(data as Record<string, unknown>);
              },
            });
          }}
        >
          {launch.isPending || preparingArtifact ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
          {preparingArtifact ? "Preparing dataset artifact" : `Launch ${METHOD_COPY[config.modality].label}`}
        </Button>
        {disabled && !launch.isPending ? (
          <div className="text-[11px] text-fg-muted">
            {disabledReason(config, preflight, artifactPreparationJob)}
          </div>
        ) : null}
        {(preflight.isError || (preflight.isSuccess && !preflight.data.ok && !preparingArtifact)) ? (
          <Button asChild variant="ghost" size="sm" className="w-full">
            <Link to="/diagnostics">Create support bundle</Link>
          </Button>
        ) : null}
      </CardContent>
    </Card>
  );
}

type PreflightCheck = {
  label: string;
  status: "ok" | "warning" | "error" | "loading" | "pending";
  detail: string;
};

function PreflightPanel({
  preflightStatus,
  checks,
}: {
  preflightStatus: "idle" | "loading" | "ok" | "error";
  checks: PreflightCheck[];
}) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardEyebrow>STATUS</CardEyebrow>
          <CardTitle>Preflight</CardTitle>
        </div>
        <Badge
          tone={preflightStatus === "ok" ? "success" : preflightStatus === "error" ? "danger" : "neutral"}
          size="sm"
        >
          {preflightStatus}
        </Badge>
      </CardHeader>
      <CardContent className="space-y-0 p-0">
        {checks.map((check) => (
          <div key={check.label} className="grid grid-cols-[20px_1fr] gap-2 border-b border-border-subtle px-4 py-3 last:border-0">
            <StatusIcon status={check.status} />
            <div className="min-w-0">
              <div className="text-[13px] font-medium text-fg">{check.label}</div>
              <div className="mt-0.5 truncate text-[12px] text-fg-muted" title={check.detail}>
                {check.detail}
              </div>
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

function LaunchSuccess({ data, payload }: { data: Record<string, unknown>; payload: Record<string, unknown> }) {
  const runId = String(data.run_id ?? data.id ?? "");
  return (
    <Card className="border-success/40">
      <CardHeader>
        <div className="flex items-center gap-2">
          <CheckCircle2 className="h-4 w-4 text-success" />
          <CardTitle>Run started</CardTitle>
        </div>
        <Badge tone="success" size="sm">running</Badge>
      </CardHeader>
      <CardContent className="space-y-3">
        <SummaryRows
          rows={[
            ["run id", runId || "-"],
            ["model", String(payload.model ?? "-")],
            [
              "source",
              String(payload.dataset_version_id ?? payload.dataset ?? payload.prompts ?? "-"),
            ],
            ["output root", String(payload.output_root ?? payload.output_dir ?? "-")],
          ]}
        />
        <div className="flex flex-wrap gap-2">
          {runId ? (
            <Button asChild size="sm" variant="primary">
              <Link to="/runs/$runId" params={{ runId }}>Open run</Link>
            </Button>
          ) : null}
          <Button asChild size="sm" variant="ghost">
            <Link to="/runs">View runs</Link>
          </Button>
          <Button asChild size="sm" variant="ghost">
              <Link to="/models" search={{ tab: "artifacts", artifact: undefined }}>Open Models when complete</Link>
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

function FormField({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="space-y-1.5">
      <Label className="text-[11px] uppercase tracking-wider text-fg-subtle">{label}</Label>
      {children}
    </div>
  );
}

function NumberField({ label, value, onChange }: { label: string; value: number; onChange: (value: number) => void }) {
  return (
    <FormField label={label}>
      <Input
        type="number"
        min={0}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </FormField>
  );
}

function StatusIcon({ status }: { status: PreflightCheck["status"] }) {
  if (status === "ok") return <CheckCircle2 className="mt-0.5 h-4 w-4 text-success" />;
  if (status === "warning") return <AlertTriangle className="mt-0.5 h-4 w-4 text-warning" />;
  if (status === "error") return <XCircle className="mt-0.5 h-4 w-4 text-danger" />;
  if (status === "loading") return <Loader2 className="mt-0.5 h-4 w-4 animate-spin text-fg-subtle" />;
  return <CircleDashed className="mt-0.5 h-4 w-4 text-fg-disabled" />;
}

function SummaryRows({ rows }: { rows: Array<[string, string]> }) {
  return (
    <dl className="divide-y divide-border-subtle rounded-md border border-border-subtle bg-bg-subtle/40">
      {rows.map(([label, value]) => (
        <div key={label} className="grid grid-cols-[86px_1fr] gap-2 px-3 py-1.5">
          <dt className="text-[10.5px] uppercase tracking-wider text-fg-disabled">{label}</dt>
          <dd className="truncate font-mono text-[11px] text-fg-subtle" title={value}>{value}</dd>
        </div>
      ))}
    </dl>
  );
}

function buildLaunchPayload(c: ConfigState, runRoot?: string): Record<string, unknown> {
  const root = runRoot || "~/.halo-forge/runs";
  const outputDir = `${root}/${c.modality}-${c.templateId ?? c.goal}-${slug(c.model || "model")}`;
  const isCustom = c.dataset === "__custom__";
  const source = c.datasetVersionId
    ? undefined
    : isCustom
      ? c.customDatasetFile
      : c.dataset;
  const lr = parseFloat(c.learningRate);
  const beta = parseFloat(c.beta);
  const rewardThreshold = parseFloat(c.rewardThreshold);
  const accelerator = c.accelerator === "mlx" || isMlxModel(c.model) ? "mlx" : undefined;
  const common = {
    mode: c.modality,
    model: c.model,
    output_dir: outputDir,
    output_root: root,
    accelerator,
    seed: c.seed,
    no_caffeinate: true,
    dataset_version_id: c.datasetVersionId || undefined,
    dataset_split: c.datasetVersionId ? c.datasetSplit || "train" : undefined,
    dataset_bindings: c.datasetBindings.length ? c.datasetBindings : undefined,
    parent_run_id: c.parentRunId || undefined,
    source_reward_integrity_audit_id: c.forkRewardAuditId || undefined,
    source_reward_integrity_decision_id: c.forkRewardDecisionId || undefined,
    fork_checkpoint_hash: c.forkCheckpointHash || undefined,
    fork_checkpoint_path: c.forkCheckpointPath || undefined,
    fork_checkpoint_occurrence_id: c.forkCheckpointOccurrenceId || undefined,
    fork_checkpoint_snapshot_path: c.forkCheckpointSnapshotPath || undefined,
    fork_boundary_unit: c.forkBoundaryUnit || undefined,
    fork_boundary_value: c.forkRewardAuditId ? c.forkBoundaryValue : undefined,
    fork_resume_mode: c.forkResumeMode || undefined,
    verifier_profile_revision_id: supportsVerifierBinding(c.modality) ? c.verifierProfileRevisionId || undefined : undefined,
    reward_system_revision_id: c.rewardAudit.enabled ? c.rewardAudit.rewardSystemRevisionId || undefined : undefined,
    reward_audit_protocol_revision_id: c.rewardAudit.enabled ? c.rewardAudit.auditProtocolRevisionId || undefined : undefined,
    reward_integrity_profile_revision_id: c.rewardAudit.enabled ? c.rewardAudit.integrityProfileRevisionId || undefined : undefined,
    reward_audit_boundaries: c.rewardAudit.enabled ? parseAuditBoundaries(c.rewardAudit.auditBoundaries) : undefined,
    development_suite_revision_id: c.rewardAudit.enabled ? c.rewardAudit.developmentSuiteRevisionId || undefined : undefined,
  };

  if (c.modality === "sft") {
    return stripEmpty({
      ...common,
      dataset: source,
      epochs: c.epochs,
      batch_size: c.batchSize,
      learning_rate: Number.isFinite(lr) ? lr : undefined,
      max_samples: c.maxSamples,
    });
  }
  if (c.modality === "raft") {
    return stripEmpty({
      ...common,
      prompts: source ? resolveRaftPrompts(source) : undefined,
      verifier: c.verifierProfileRevisionId ? undefined : c.verifier,
      cycles: c.cycles,
      samples_per_prompt: c.samplesPerPrompt,
      keep_percent: 0.5,
      reward_threshold: 0.5,
    });
  }
  if (c.modality === "dpo") {
    return stripEmpty({
      ...common,
      dataset: source,
      epochs: c.epochs,
      batch_size: c.batchSize,
      learning_rate: Number.isFinite(lr) ? lr : undefined,
      max_samples: c.maxSamples,
      beta: Number.isFinite(beta) ? beta : undefined,
      loss_type: c.lossType,
      reference_free: c.referenceFree,
    });
  }
  if (c.modality === "orpo") {
    return stripEmpty({
      ...common,
      dataset: source,
      epochs: c.epochs,
      batch_size: c.batchSize,
      learning_rate: Number.isFinite(lr) ? lr : undefined,
      max_samples: c.maxSamples,
      beta: Number.isFinite(beta) ? beta : undefined,
    });
  }
  if (c.modality === "rm") {
    return stripEmpty({
      ...common,
      dataset: source,
      epochs: c.epochs,
      batch_size: c.batchSize,
      learning_rate: Number.isFinite(lr) ? lr : undefined,
      max_samples: c.maxSamples,
    });
  }
  if (c.modality === "grpo") {
    return stripEmpty({
      ...common,
      dataset: source,
      epochs: c.epochs,
      batch_size: c.batchSize,
      learning_rate: Number.isFinite(lr) ? lr : undefined,
      max_samples: c.maxSamples,
      beta: Number.isFinite(beta) ? beta : undefined,
      reference_free: c.referenceFree,
      verifier: c.verifierProfileRevisionId ? undefined : c.verifier,
      num_generations: c.numGenerations,
      epsilon: 0.2,
      temperature: 0.9,
      reward_threshold: Number.isFinite(rewardThreshold) ? rewardThreshold : 0,
    });
  }
  if (["classify", "embed", "rerank"].includes(c.modality)) {
    return stripEmpty({
      ...common,
      dataset: source,
      epochs: c.epochs,
      batch_size: c.batchSize,
      learning_rate: Number.isFinite(lr) ? lr : undefined,
      max_samples: c.maxSamples,
    });
  }
  return stripEmpty({
    ...common,
    dataset: source,
    cycles: c.cycles,
    samples_per_prompt: c.modality === "vlm" || c.modality === "audio" ? c.samplesPerPrompt : undefined,
    limit: c.modality === "vlm" || c.modality === "reasoning" || c.modality === "agentic" ? c.maxSamples : undefined,
    keep_percent: c.modality === "vlm" || c.modality === "audio" ? 0.5 : undefined,
    reward_threshold: c.modality === "vlm" || c.modality === "audio" ? 0.5 : undefined,
    task: c.modality === "audio" ? c.task : undefined,
    learning_rate: c.modality === "reasoning" || c.modality === "agentic" ? lr : undefined,
    allow_prototype_train: c.allowPrototypeTrain,
  });
}

function buildPreflightChecks(
  config: ConfigState,
  preflight: ReturnType<typeof useTrainingPreflight>,
  backendName: string | undefined,
  mlxReadiness: BackendInfo["mlx_readiness"] | undefined,
  artifactPreparationJob?: DatasetJob,
): PreflightCheck[] {
  const wantsMlx = config.accelerator === "mlx" || isMlxModel(config.model);
  const checks: PreflightCheck[] = [
    {
      label: "Backend connected",
      status: backendName ? "ok" : "loading",
      detail: backendName ? `Active accelerator: ${backendName}` : "Detecting...",
    },
    {
      label: "Model identifier set",
      status: config.model ? "ok" : "pending",
      detail: config.model || "Type a HuggingFace or MLX repo id above",
    },
    {
      label: sourceLabel(config.modality),
      status:
        config.datasetVersionId
          ? "ok"
          : config.dataset === "__custom__"
          ? config.customDatasetFile.trim()
            ? "ok"
            : "warning"
          : config.dataset
            ? "ok"
            : "pending",
      detail:
        config.datasetVersionId
          ? `${config.datasetVersionId} · ${config.datasetSplit || "train"}`
          : config.dataset === "__custom__"
          ? config.customDatasetFile || "Provide a path to a JSONL file"
          : config.dataset,
    },
  ];

  if (wantsMlx) {
    checks.push({
      label: "MLX readiness",
      status: mlxReadiness?.executable ? "ok" : mlxReadiness ? "warning" : "loading",
      detail: mlxReadiness?.executable
        ? "MLX executable probe passed"
        : mlxReadiness?.errors?.[0] ?? mlxReadiness?.warnings?.[0] ?? "Checking MLX runtime",
    });
  }

  if (config.modality === "raft" || config.modality === "grpo") {
    checks.push({
      label: "Verifier identity",
      status: hasVerifier(config) ? "ok" : "pending",
      detail: config.verifierProfileRevisionId
        ? `Qualified revision · ${config.verifierProfileRevisionId}`
        : config.verifier
          ? `Legacy unqualified · ${config.verifier}`
          : "Pick a verifier",
    });
  }

  if (config.rewardAudit.enabled) {
    checks.push({
      label: "Training signal audit",
      status: rewardAuditReady(config.rewardAudit) ? "ok" : "pending",
      detail: rewardAuditReady(config.rewardAudit)
        ? `Same-output audit · ${config.rewardAudit.auditBoundaries || "resolved boundaries"}`
        : "Choose a reward system, capture protocol, and integrity policy",
    });
  }

  if (["vlm", "audio", "reasoning", "agentic"].includes(config.modality) && !config.allowPrototypeTrain) {
    checks.push({
      label: "Capability gate",
      status: "warning",
      detail: "Some model families may need the prototype train gate in Advanced.",
    });
  }

  if (
    preflight.isSuccess &&
    (preflight.data.status === "preparing_dataset" || preflight.data.ready === false)
  ) {
    const failed =
      artifactPreparationJob &&
      ["failed", "cancelled"].includes(artifactPreparationJob.status);
    const progress =
      artifactPreparationJob?.progress_percent ??
      preflight.data.artifact_preparation?.progress_percent;
    const stage =
      artifactPreparationJob?.stage ?? preflight.data.artifact_preparation?.stage;
    checks.push({
      label: "Training dataset artifact",
      status: failed ? "error" : "loading",
      detail: failed
        ? artifactPreparationJob?.error ?? "Artifact preparation did not complete"
        : `${stage || "queued"}${typeof progress === "number" ? ` · ${progress.toFixed(0)}%` : ""}`,
    });
  } else if (preflight.isPending) {
    checks.push({ label: "Server preflight", status: "loading", detail: "Validating launch..." });
  } else if (preflight.isError) {
    checks.push({ label: "Server preflight", status: "error", detail: (preflight.error as Error).message });
  } else if (preflight.isSuccess && preflight.data) {
    const issue = preflight.data.errors[0] ?? preflight.data.suggested_fixes[0];
    checks.push({
      label: "Server preflight",
      status: preflight.data.ok ? (preflight.data.warnings.length ? "warning" : "ok") : "error",
      detail: issue ?? preflight.data.user_summary?.headline ?? "All checks passed",
    });
  } else {
    checks.push({ label: "Server preflight", status: "pending", detail: "Runs automatically as you edit the form" });
  }
  return checks;
}

function applyTemplate(prev: ConfigState, template: TrainingTemplate): ConfigState {
  if (!isTrainingMode(template.modality)) return prev;
  const hp = template.hyperparams;
  const goal = goalForTemplate(template);
  const next = withModeDefaults({ ...prev, goal }, template.modality);
  return {
    ...next,
    templateId: template.id,
    model: template.model_hint || next.model,
    dataset: template.dataset_hint && template.dataset_hint !== "@custom"
      ? normalizeSourceForMode(template.modality, template.dataset_hint)
      : template.dataset_hint === "@custom"
        ? "__custom__"
        : next.dataset,
    verifier: typeof template.verifier === "string" ? template.verifier : next.verifier,
    verifierProfileRevisionId: "",
    rewardAudit: EMPTY_REWARD_AUDIT_BINDING,
    epochs: numberFrom(hp.epochs, next.epochs),
    batchSize: numberFrom(hp.batch_size, next.batchSize),
    learningRate: stringFrom(hp.learning_rate, next.learningRate),
    seed: numberFrom(hp.seed, next.seed),
    cycles: numberFrom(hp.cycles, next.cycles),
    samplesPerPrompt: numberFrom(hp.samples_per_prompt, next.samplesPerPrompt),
    maxSamples: numberFrom(hp.max_samples, next.maxSamples),
    beta: stringFrom(hp.beta, next.beta),
    lossType: typeof hp.loss_type === "string" ? hp.loss_type : next.lossType,
    numGenerations: numberFrom(hp.group_size ?? hp.num_generations, next.numGenerations),
    accelerator: hp.accelerator === "mlx" || isMlxModel(template.model_hint) ? "mlx" : next.accelerator,
  };
}

function withModeDefaults(config: ConfigState, modality: TrainingMode): ConfigState {
  const defaults = DEFAULTS[modality] ?? {};
  return {
    ...config,
    ...defaults,
    modality,
    templateId: null,
    verifierProfileRevisionId: config.modality === modality ? config.verifierProfileRevisionId : "",
    rewardAudit: config.modality === modality ? config.rewardAudit : EMPTY_REWARD_AUDIT_BINDING,
    accelerator: config.accelerator === "mlx" ? "mlx" : (defaults.accelerator as Accelerator | undefined) ?? "auto",
  };
}

function sourcesForMode(mode: TrainingMode, datasets: TrainingSource[]): TrainingSource[] {
  if (mode === "sft") return datasets.length ? datasets : [{ key: "codealpaca", description: "Code instruction data.", size_hint: "small", domain: "code" }];
  return MODALITY_SOURCES[mode] ?? [];
}

function sourceLabel(mode: TrainingMode): string {
  if (mode === "raft") return "Prompt source";
  if (mode === "grpo") return "Prompt dataset";
  if (preferenceMode(mode) || mode === "rm") return "Preference dataset";
  return "Dataset";
}

function supportsVerifierBinding(mode: TrainingMode): boolean {
  return ["raft", "grpo", "reasoning", "agentic", "vlm", "audio"].includes(mode);
}

function hasVerifier(config: ConfigState): boolean {
  return Boolean(config.verifierProfileRevisionId || config.verifier);
}

function isVerifierModalityCompatible(verifierModality: string, trainingMode: TrainingMode): boolean {
  const normalized = verifierModality.toLowerCase().replace(/[-\s]/g, "_");
  if (trainingMode === "vlm") return ["vlm", "vision", "image", "multimodal"].includes(normalized);
  if (trainingMode === "audio") return ["audio", "speech"].includes(normalized);
  if (trainingMode === "agentic") return ["agentic", "tool", "tool_use", "text"].includes(normalized);
  return ["text", "reasoning", trainingMode].includes(normalized);
}

function cycleMode(mode: TrainingMode): boolean {
  return ["raft", "vlm", "audio", "reasoning", "agentic"].includes(mode);
}

function preferenceMode(mode: TrainingMode): boolean {
  return mode === "dpo" || mode === "orpo";
}

function goalLabel(goal: GoalKey): string {
  return GOALS.find((item) => item.key === goal)?.label ?? goal;
}

function goalForTemplate(template: TrainingTemplate): GoalKey {
  if (template.category === "vision") return "vision";
  if (template.category === "audio") return "audio";
  if (template.category === "preference") return "preferences";
  if (template.category === "agentic") return "tool-use";
  if (template.category === "reasoning") return "reasoning";
  return "code";
}

function resolveRaftPrompts(source: string): string {
  return RAFT_PROMPT_ALIASES[source] ?? source;
}

function normalizeSourceForMode(mode: TrainingMode, source: string): string {
  if (mode === "raft") return resolveRaftPrompts(source);
  return source;
}

function stripEmpty(o: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(o)) {
    if (v === undefined || v === null || v === "") continue;
    if (typeof v === "boolean" && v === false) continue;
    out[k] = v;
  }
  return out;
}

function canLaunch(c: ConfigState): boolean {
  if (!c.model.trim()) return false;
  if (c.rewardAudit.enabled && !rewardAuditReady(c.rewardAudit)) return false;
  if (c.datasetVersionId) return !(c.modality === "grpo" && !hasVerifier(c)) && !(c.modality === "audio" && !c.task);
  if (c.dataset === "__custom__" && !c.customDatasetFile.trim()) return false;
  if (!c.dataset) return false;
  if (c.modality === "grpo" && !hasVerifier(c)) return false;
  if (c.modality === "audio" && !c.task) return false;
  return true;
}

function rewardAuditReady(value: RewardAuditBindingValue): boolean {
  return Boolean(value.rewardSystemRevisionId && value.auditProtocolRevisionId && value.integrityProfileRevisionId);
}

function parseAuditBoundaries(value: string): Array<number | string> | undefined {
  const items = value.split(",").map((part) => part.trim()).filter(Boolean).slice(0, 4).map((part) => {
    if (part.endsWith("%")) return part;
    const numeric = Number(part);
    return Number.isFinite(numeric) && numeric > 0 ? numeric : part;
  });
  return items.length ? items : undefined;
}

function disabledReason(
  c: ConfigState,
  preflight: ReturnType<typeof useTrainingPreflight>,
  artifactPreparationJob?: DatasetJob,
): string {
  if (!c.model.trim()) return "Choose or type a base model before launching.";
  if (c.rewardAudit.enabled && !rewardAuditReady(c.rewardAudit)) return "Complete the reward system, capture protocol, and integrity policy before launching an audited run.";
  if (c.datasetVersionId) {
    if (c.modality === "grpo" && !hasVerifier(c)) return "Choose a verifier before launching GRPO.";
    if (c.modality === "audio" && !c.task) return "Choose the audio task before launching.";
    if (
      preflight.isSuccess &&
      (preflight.data.status === "preparing_dataset" || preflight.data.ready === false)
    ) {
      if (artifactPreparationJob?.status === "failed") {
        return artifactPreparationJob.error ?? "Dataset artifact preparation failed.";
      }
      if (artifactPreparationJob?.status === "cancelled") {
        return "Dataset artifact preparation was cancelled.";
      }
      return "Rendering and validating the immutable trainer artifact.";
    }
    if (preflight.isSuccess && !preflight.data.ok) {
      return preflight.data.suggested_fixes[0] ?? "Resolve the preflight issue above before launching.";
    }
    if (!preflight.isSuccess) return "Waiting for server preflight.";
    return "Waiting for launch requirements.";
  }
  if (!c.dataset) return "Choose a dataset or source before launching.";
  if (c.dataset === "__custom__" && !c.customDatasetFile.trim()) return "Add the local JSONL path for the custom dataset.";
  if (c.modality === "grpo" && !hasVerifier(c)) return "Choose a verifier before launching GRPO.";
  if (c.modality === "audio" && !c.task) return "Choose the audio task before launching.";
  if (preflight.isSuccess && !preflight.data.ok) {
    return preflight.data.suggested_fixes[0] ?? "Resolve the preflight issue above before launching.";
  }
  return "Waiting for launch requirements.";
}

function launchHint(mode: TrainingMode): { headline: string; detail: string } {
  if (mode === "sft") {
    return {
      headline: "What happens next",
      detail: "Halo Forge starts a conservative supervised run and keeps you here with links to the run monitor.",
    };
  }
  if (mode === "raft") {
    return {
      headline: "What happens next",
      detail: "Halo Forge samples answers, checks them with the verifier, and trains on the kept examples.",
    };
  }
  if (mode === "dpo" || mode === "orpo") {
    return {
      headline: "What happens next",
      detail: "Halo Forge launches preference tuning with the selected pair dataset and writes a complete launch context.",
    };
  }
  if (mode === "rm") {
    return {
      headline: "What happens next",
      detail: "Halo Forge trains a reward scorer from chosen/rejected pairs and records the output for later comparison.",
    };
  }
  if (mode === "grpo") {
    return {
      headline: "What happens next",
      detail: "Halo Forge runs verifier-grounded RL. Start small and watch the run monitor for reward and verifier signals.",
    };
  }
  return {
    headline: "What happens next",
    detail: "Halo Forge launches the method-specific trainer and records artifacts under the workstation run folder.",
  };
}

function preflightStatus(
  preflight: ReturnType<typeof useTrainingPreflight>,
  config: ConfigState,
  artifactPreparationJob?: DatasetJob,
): "idle" | "loading" | "ok" | "error" {
  if (!canLaunch(config)) return "idle";
  if (preflight.isPending) return "loading";
  if (preflight.isError) return "error";
  if (preflight.isSuccess) {
    if (
      preflight.data.status === "preparing_dataset" ||
      preflight.data.ready === false
    ) {
      return artifactPreparationJob &&
        ["failed", "cancelled"].includes(artifactPreparationJob.status)
        ? "error"
        : "loading";
    }
    return preflight.data.ok ? "ok" : "error";
  }
  return "idle";
}

function isTrainingMode(value: unknown): value is TrainingMode {
  return typeof value === "string" && TRAINING_MODES.includes(value as TrainingMode);
}

function isGoalKey(value: unknown): value is GoalKey {
  return typeof value === "string" && GOALS.some((goal) => goal.key === value);
}

function isMlxModel(model: string | undefined): boolean {
  return Boolean(model && model.startsWith("mlx-community/"));
}

function upsertBinding(bindings: DatasetBinding[], next: DatasetBinding): DatasetBinding[] {
  return [...bindings.filter((binding) => binding.role !== next.role), next];
}

function applyResolvedLaunchConfig(
  previous: ConfigState,
  parentRunId: string,
  raw: Record<string, unknown>,
  recordedBindings: DatasetBinding[],
): ConfigState {
  const mode = isTrainingMode(raw.mode) ? raw.mode : previous.modality;
  const rawBindings = Array.isArray(raw.dataset_bindings)
    ? raw.dataset_bindings.filter(isDatasetBinding)
    : [];
  const bindings = recordedBindings.length ? recordedBindings : rawBindings;
  const train = bindings.find((binding) => binding.role === "train");
  const source = stringValue(raw.dataset ?? raw.prompts, previous.dataset);
  return {
    ...withModeDefaults(previous, mode),
    parentRunId,
    goal: goalForMode(mode),
    modality: mode,
    model: stringValue(raw.model, previous.model),
    dataset: source || previous.dataset,
    customDatasetFile: source.startsWith("/") || source.endsWith(".jsonl") ? source : previous.customDatasetFile,
    datasetVersionId: train?.dataset_version_id ?? stringValue(raw.dataset_version_id, ""),
    datasetSplit: train?.split ?? stringValue(raw.dataset_split, "train"),
    datasetBindings: bindings.length
      ? bindings
      : raw.dataset_version_id
        ? [{ role: "train", dataset_version_id: String(raw.dataset_version_id), split: stringValue(raw.dataset_split, "train") }]
        : [],
    accelerator: raw.accelerator === "mlx" ? "mlx" : previous.accelerator,
    verifier: stringValue(raw.verifier, previous.verifier),
    verifierProfileRevisionId: stringValue(raw.verifier_profile_revision_id, ""),
    rewardAudit: {
      enabled: Boolean(raw.reward_system_revision_id),
      rewardSystemRevisionId: stringValue(raw.reward_system_revision_id, ""),
      auditProtocolRevisionId: stringValue(raw.reward_audit_protocol_revision_id, ""),
      integrityProfileRevisionId: stringValue(raw.reward_integrity_profile_revision_id, ""),
      auditBoundaries: Array.isArray(raw.reward_audit_boundaries) ? raw.reward_audit_boundaries.join(", ") : "",
      developmentSuiteRevisionId: stringValue(raw.development_suite_revision_id, ""),
    },
    task: stringValue(raw.task, previous.task),
    epochs: finiteNumber(raw.epochs, previous.epochs),
    batchSize: finiteNumber(raw.batch_size, previous.batchSize),
    learningRate: stringValue(raw.learning_rate, previous.learningRate),
    seed: finiteNumber(raw.seed, previous.seed),
    cycles: finiteNumber(raw.cycles, previous.cycles),
    samplesPerPrompt: finiteNumber(raw.samples_per_prompt, previous.samplesPerPrompt),
    maxSamples: finiteNumber(raw.max_samples ?? raw.limit, previous.maxSamples),
    beta: stringValue(raw.beta, previous.beta),
    lossType: stringValue(raw.loss_type, previous.lossType),
    referenceFree: raw.reference_free === true,
    numGenerations: finiteNumber(raw.num_generations ?? raw.group_size, previous.numGenerations),
    rewardThreshold: stringValue(raw.reward_threshold, previous.rewardThreshold),
    allowPrototypeTrain: raw.allow_prototype_train === true,
    templateId: typeof raw.template_id === "string" ? raw.template_id : null,
  };
}

function applyRewardAuditForkContext(
  previous: ConfigState,
  context: RewardIntegrityForkContext,
): ConfigState {
  const resolved = applyResolvedLaunchConfig(
    previous,
    context.parent_run_id,
    context.train_context,
    [],
  );
  return {
    ...resolved,
    forkRewardAuditId: context.audit_id,
    forkRewardDecisionId: context.decision.id,
    forkCheckpointHash: context.checkpoint.content_hash,
    forkCheckpointPath: context.checkpoint.path || "",
    forkCheckpointOccurrenceId: context.checkpoint.occurrence_id || "",
    forkCheckpointSnapshotPath: context.checkpoint.snapshot_path || "",
    forkBoundaryUnit: context.checkpoint.boundary_unit || "",
    forkBoundaryValue: context.checkpoint.boundary_value ?? 0,
    forkResumeMode: context.resume_mode,
  };
}

function isDatasetBinding(value: unknown): value is DatasetBinding {
  if (!value || typeof value !== "object") return false;
  const item = value as Record<string, unknown>;
  return typeof item.role === "string" && typeof item.dataset_version_id === "string" && typeof item.split === "string";
}

function goalForMode(mode: TrainingMode): GoalKey {
  if (["classify", "embed", "rerank"].includes(mode)) return "task-models";
  if (mode === "vlm") return "vision";
  if (mode === "audio") return "audio";
  if (["dpo", "orpo", "rm"].includes(mode)) return "preferences";
  if (mode === "agentic") return "tool-use";
  if (mode === "reasoning") return "reasoning";
  return "code";
}

function stringValue(value: unknown, fallback: string): string {
  return typeof value === "string" || typeof value === "number" ? String(value) : fallback;
}

function finiteNumber(value: unknown, fallback: number): number {
  const candidate = typeof value === "number" ? value : Number(value);
  return Number.isFinite(candidate) ? candidate : fallback;
}

function formatRowCount(value: number | null | undefined): string {
  return typeof value === "number" ? `${new Intl.NumberFormat().format(value)} rows` : "row count unknown";
}

function slug(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 40);
}

function numberFrom(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function stringFrom(value: unknown, fallback: string): string {
  return typeof value === "number" || typeof value === "string" ? String(value) : fallback;
}

function ReadoutItem({ label, value }: { label: string; value: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="text-fg-disabled tracking-wider">{label}</span>
      <span className="text-fg">{value}</span>
    </span>
  );
}

function ReadoutSep() {
  return <span className="text-fg-disabled">·</span>;
}
