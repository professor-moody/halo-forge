import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import {
  AlertTriangle,
  AudioLines,
  BookOpen,
  Brain,
  CheckCircle2,
  CircleDashed,
  Code2,
  Eye,
  Loader2,
  Play,
  Settings2,
  ShieldCheck,
  Sparkles,
  Wrench,
  XCircle,
  type LucideIcon,
} from "lucide-react";
import { useEffect, useMemo, useState, type ReactNode } from "react";
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
import type { BackendInfo, ModelCatalogEntry, TrainingMode, TrainingTemplate } from "@/lib/api";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/train")({
  component: TrainConfiguratorRoute,
  validateSearch: (search): { template?: string; model?: string; mode?: string } => ({
    template: typeof search.template === "string" ? search.template : undefined,
    model: typeof search.model === "string" ? search.model : undefined,
    mode: typeof search.mode === "string" ? search.mode : undefined,
  }),
});

type GoalKey = "code" | "reasoning" | "tool-use" | "vision" | "audio" | "preferences";
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
};

const METHOD_GUIDANCE: Record<TrainingMode, { headline: string; detail: string }> = {
  sft: {
    headline: "Best default when you have examples.",
    detail: "Halo Forge will train on the selected dataset and write the run under the workstation run folder.",
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
};

const DEFAULTS: Record<TrainingMode, Partial<ConfigState>> = {
  sft: { dataset: "codealpaca", epochs: 1, batchSize: 2, learningRate: "2e-4", maxSamples: 200 },
  raft: { dataset: DEFAULT_RAFT_PROMPTS, cycles: 1, samplesPerPrompt: 4, verifier: "execution" },
  dpo: { dataset: "ultrafeedback", epochs: 1, batchSize: 1, learningRate: "5e-6", beta: "0.1", lossType: "sigmoid" },
  orpo: { dataset: "ultrafeedback", epochs: 1, batchSize: 1, learningRate: "8e-6", beta: "0.1" },
  rm: { dataset: "ultrafeedback", epochs: 1, batchSize: 4, learningRate: "1e-5" },
  grpo: { dataset: "gsm8k", epochs: 1, batchSize: 1, learningRate: "1e-6", beta: "0.04", verifier: "json_schema", numGenerations: 4 },
  vlm: { dataset: "textvqa", cycles: 1, samplesPerPrompt: 2, maxSamples: 24 },
  audio: { dataset: "librispeech", cycles: 1, samplesPerPrompt: 2, task: "asr" },
  reasoning: { dataset: "gsm8k", cycles: 1, maxSamples: 64, learningRate: "1e-5" },
  agentic: { dataset: "xlam_sft", cycles: 1, maxSamples: 64, learningRate: "5e-5" },
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
};

interface ConfigState {
  goal: GoalKey;
  modality: TrainingMode;
  model: string;
  dataset: string;
  customDatasetFile: string;
  accelerator: Accelerator;
  verifier: string;
  task: string;
  epochs: number;
  batchSize: number;
  learningRate: string;
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
    accelerator: "auto",
    verifier: "execution",
    task: "asr",
    epochs: 1,
    batchSize: 2,
    learningRate: "2e-4",
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
  const workspace = useWorkspaceInfo();
  const datasets = useTrainingDatasets();
  const verifiers = useTrainingVerifiers();
  const preflight = useTrainingPreflight();
  const launch = useTrainingLaunch();
  const { template: templateId, model: modelId, mode } = Route.useSearch();

  const [config, setConfig] = useState<ConfigState>(defaultConfig);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [launchedRun, setLaunchedRun] = useState<Record<string, unknown> | null>(null);
  const models = useTrainingModels({ mode: config.modality });
  const mlxModels = useModelCatalog({ mode: config.modality, backend: "mlx" });
  const mlxReadiness = backend.data?.mlx_readiness;
  const mlxReady = mlxReadiness?.executable === true;
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
  const currentPreflightStatus = preflightStatus(preflight, config);
  const payload = useMemo(
    () => buildLaunchPayload(config, workspace.data?.default_run_root),
    [config, workspace.data?.default_run_root],
  );

  const templateQuery = useQuery({
    queryKey: ["training-template", templateId],
    queryFn: () => api.trainingTemplate(templateId!),
    enabled: Boolean(templateId),
  });

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
    config.accelerator,
    config.verifier,
    config.task,
    config.epochs,
    config.batchSize,
    config.learningRate,
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

  return (
    <>
      <Topbar
        eyebrow="Workspace"
        title="Train"
        subtitle="Choose a goal and method; Halo Forge generates a conservative launch you can inspect."
        actions={
          <>
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
              modelSuggestions={modelSuggestions}
              selectedModel={selectedModel}
              mlxReady={mlxReady}
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
              checks={buildPreflightChecks(config, preflight, backend.data?.name, mlxReadiness)}
            />
            <LaunchPanel
              config={config}
              payload={payload}
              selectedModel={selectedModel}
              preflight={preflight}
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
  modelSuggestions,
  selectedModel,
  mlxReady,
}: {
  config: ConfigState;
  setConfig: (updater: (c: ConfigState) => ConfigState) => void;
  datasets: TrainingSource[];
  verifiers: Array<{ key: string; label: string; toolchain: string }>;
  modelSuggestions: ModelCatalogEntry[];
  selectedModel: ModelCatalogEntry | null;
  mlxReady: boolean;
}) {
  const sources = sourcesForMode(config.modality, datasets);
  const isCustom = config.dataset === "__custom__";
  const needsVerifier = config.modality === "raft" || config.modality === "grpo";

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

        <div className="grid gap-3 lg:grid-cols-2">
          <FormField label={sourceLabel(config.modality)}>
            <Select
              value={config.dataset}
              onValueChange={(value) => setConfig((prev) => ({ ...prev, dataset: value }))}
            >
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                {sources.map((source) => (
                  <SelectItem key={source.key} value={source.key}>
                    {source.key}
                  </SelectItem>
                ))}
                <SelectItem value="__custom__">Custom local file</SelectItem>
              </SelectContent>
            </Select>
          </FormField>
          {isCustom ? (
            <FormField label="Local JSONL path">
              <Input
                value={config.customDatasetFile}
                onChange={(event) => setConfig((prev) => ({ ...prev, customDatasetFile: event.target.value }))}
                placeholder="/path/to/training.jsonl"
              />
            </FormField>
          ) : (
            <FormField label="Dataset note">
              <div className="min-h-9 rounded-md border border-border bg-bg-subtle px-3 py-2 text-[12px] text-fg-muted">
                {sources.find((source) => source.key === config.dataset)?.description ?? "Registered dataset or source."}
              </div>
            </FormField>
          )}
        </div>

        {needsVerifier ? (
          <div className="grid gap-3 lg:grid-cols-2">
            <FormField label="Verifier">
              <Select
                value={config.verifier}
                onValueChange={(value) => setConfig((prev) => ({ ...prev, verifier: value }))}
              >
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {verifiers.map((verifier) => (
                    <SelectItem key={verifier.key} value={verifier.key}>
                      {verifier.key}
                    </SelectItem>
                  ))}
                  <SelectItem value="json_schema">json_schema</SelectItem>
                  <SelectItem value="llm_judge">llm_judge</SelectItem>
                </SelectContent>
              </Select>
            </FormField>
            <FormField label="Verifier note">
              <div className="min-h-9 rounded-md border border-border bg-bg-subtle px-3 py-2 text-[12px] text-fg-muted">
                {config.modality === "grpo"
                  ? "GRPO uses the verifier as the reward function."
                  : "RAFT verifies generations before training on kept samples."}
              </div>
            </FormField>
          </div>
        ) : null}

        {config.modality === "audio" ? (
          <FormField label="Audio task">
            <Select
              value={config.task}
              onValueChange={(value) => setConfig((prev) => ({ ...prev, task: value }))}
            >
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="asr">ASR</SelectItem>
                <SelectItem value="classification">Classification</SelectItem>
                <SelectItem value="tts">TTS</SelectItem>
              </SelectContent>
            </Select>
          </FormField>
        ) : null}
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
  launch,
  onLaunched,
}: {
  config: ConfigState;
  payload: Record<string, unknown>;
  selectedModel: ModelCatalogEntry | null;
  preflight: ReturnType<typeof useTrainingPreflight>;
  launch: ReturnType<typeof useTrainingLaunch>;
  onLaunched: (data: Record<string, unknown>) => void;
}) {
  const disabled = !canLaunch(config) || launch.isPending || (preflight.isSuccess && !preflight.data.ok);
  const launchCopy = launchHint(config.modality);
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardEyebrow>LAUNCH</CardEyebrow>
          <CardTitle>Launch summary</CardTitle>
        </div>
        <Badge tone={preflight.isSuccess && preflight.data.ok ? "success" : "neutral"} size="sm">
          {config.modality}
        </Badge>
      </CardHeader>
      <CardContent className="space-y-3">
        <SummaryRows
          rows={[
            ["method", config.modality],
            ["model", String(payload.model ?? "-")],
            [sourceLabel(config.modality).toLowerCase(), String(payload.dataset ?? payload.prompts ?? "-")],
            ["output", String(payload.output_dir ?? "-")],
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
            launch.mutate(payload, {
              onSuccess: (data) => onLaunched(data as Record<string, unknown>),
            });
          }}
        >
          {launch.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
          Launch {METHOD_COPY[config.modality].label}
        </Button>
        {disabled && !launch.isPending ? (
          <div className="text-[11px] text-fg-muted">
            {disabledReason(config, preflight)}
          </div>
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
            ["source", String(payload.dataset ?? payload.prompts ?? "-")],
            ["output", String(payload.output_dir ?? "-")],
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
            <Link to="/results">Serve when complete</Link>
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
  const source = isCustom ? c.customDatasetFile : c.dataset;
  const lr = parseFloat(c.learningRate);
  const beta = parseFloat(c.beta);
  const rewardThreshold = parseFloat(c.rewardThreshold);
  const accelerator = c.accelerator === "mlx" || isMlxModel(c.model) ? "mlx" : undefined;
  const common = {
    mode: c.modality,
    model: c.model,
    output_dir: outputDir,
    accelerator,
    no_caffeinate: true,
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
      prompts: resolveRaftPrompts(source),
      verifier: c.verifier,
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
      verifier: c.verifier,
      num_generations: c.numGenerations,
      epsilon: 0.2,
      temperature: 0.9,
      reward_threshold: Number.isFinite(rewardThreshold) ? rewardThreshold : 0,
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
        config.dataset === "__custom__"
          ? config.customDatasetFile.trim()
            ? "ok"
            : "warning"
          : config.dataset
            ? "ok"
            : "pending",
      detail:
        config.dataset === "__custom__"
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
      label: "Verifier toolchain",
      status: config.verifier ? "ok" : "pending",
      detail: config.verifier || "Pick a verifier",
    });
  }

  if (["vlm", "audio", "reasoning", "agentic"].includes(config.modality) && !config.allowPrototypeTrain) {
    checks.push({
      label: "Capability gate",
      status: "warning",
      detail: "Some model families may need the prototype train gate in Advanced.",
    });
  }

  if (preflight.isPending) {
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
    epochs: numberFrom(hp.epochs, next.epochs),
    batchSize: numberFrom(hp.batch_size, next.batchSize),
    learningRate: stringFrom(hp.learning_rate, next.learningRate),
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
  if (c.dataset === "__custom__" && !c.customDatasetFile.trim()) return false;
  if (!c.dataset) return false;
  if (c.modality === "grpo" && !c.verifier) return false;
  if (c.modality === "audio" && !c.task) return false;
  return true;
}

function disabledReason(
  c: ConfigState,
  preflight: ReturnType<typeof useTrainingPreflight>,
): string {
  if (!c.model.trim()) return "Choose or type a base model before launching.";
  if (!c.dataset) return "Choose a dataset or source before launching.";
  if (c.dataset === "__custom__" && !c.customDatasetFile.trim()) return "Add the local JSONL path for the custom dataset.";
  if (c.modality === "grpo" && !c.verifier) return "Choose a verifier before launching GRPO.";
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
): "idle" | "loading" | "ok" | "error" {
  if (!canLaunch(config)) return "idle";
  if (preflight.isPending) return "loading";
  if (preflight.isError) return "error";
  if (preflight.isSuccess) return preflight.data.ok ? "ok" : "error";
  return "idle";
}

function isTrainingMode(value: unknown): value is TrainingMode {
  return typeof value === "string" && TRAINING_MODES.includes(value as TrainingMode);
}

function isMlxModel(model: string | undefined): boolean {
  return Boolean(model && model.startsWith("mlx-community/"));
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
