import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import {
  AlertTriangle,
  BookOpen,
  CheckCircle2,
  ChevronRight,
  CircleDashed,
  ClipboardList,
  FileQuestion,
  Loader2,
  Play,
  Settings2,
  Sparkles,
  XCircle,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Topbar } from "@/components/shell";
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
  useTrainingDatasets,
  useTrainingLaunch,
  useTrainingModels,
  useTrainingPreflight,
  useTrainingVerifiers,
} from "@/lib/hooks";
import type { ModelCatalogEntry } from "@/lib/api";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/train")({
  component: TrainConfiguratorRoute,
  validateSearch: (search): { template?: string; model?: string; mode?: string } => ({
    template: typeof search.template === "string" ? search.template : undefined,
    model: typeof search.model === "string" ? search.model : undefined,
    mode: typeof search.mode === "string" ? search.mode : undefined,
  }),
});

/* -------------------------------------------------------------------------
 * Configurator state.
 *
 * Single-page, single-form design. Stripe / Vercel pattern: every
 * setting is visible at once, with the preflight panel on the right
 * giving live feedback. No wizard transitions — engineering tools
 * don't make you click "Next →" through dialogs.
 * ----------------------------------------------------------------------- */

type Modality = "sft" | "raft";

interface ConfigState {
  modality: Modality;
  model: string;
  dataset: string;
  customDatasetFile: string; // when dataset === "__custom__"
  verifier: string; // RAFT only
  // hyperparameters
  epochs: number;
  batchSize: number;
  learningRate: string; // string so the user can type "2e-4" without coercion fights
  loraRank: number;
  loraAlpha: number;
  maxSeqLength: number;
  cycles: number; // RAFT only
  samplesPerPrompt: number; // RAFT only
}

function defaultConfig(): ConfigState {
  return {
    modality: "sft",
    model: "",
    dataset: "codealpaca",
    customDatasetFile: "",
    verifier: "gcc",
    epochs: 3,
    batchSize: 2,
    learningRate: "2e-4",
    loraRank: 16,
    loraAlpha: 32,
    maxSeqLength: 2048,
    cycles: 3,
    samplesPerPrompt: 8,
  };
}

function TrainConfiguratorRoute() {
  const backend = useBackendInfo();
  const datasets = useTrainingDatasets();
  const verifiers = useTrainingVerifiers();
  const preflight = useTrainingPreflight();
  const launch = useTrainingLaunch();
  const { template: templateId, model: modelId, mode } = Route.useSearch();

  const [config, setConfig] = useState<ConfigState>(defaultConfig);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [templateApplied, setTemplateApplied] = useState<string | null>(null);
  const models = useTrainingModels({ mode: config.modality });
  const selectedModel = useMemo(
    () => models.data?.items.find((item) => item.id === config.model) ?? null,
    [config.model, models.data?.items],
  );
  const currentPreflightStatus = preflightStatus(preflight, config);

  useEffect(() => {
    if (!modelId) return;
    setConfig((prev) => ({
      ...prev,
      model: modelId,
      modality: mode === "raft" || mode === "sft" ? mode : prev.modality,
    }));
  }, [modelId, mode]);

  // Pull the template if the user landed via /train?template=<id>.
  // Only fetch on demand to keep the no-template path zero-cost.
  const templateQuery = useQuery({
    queryKey: ["training-template", templateId],
    queryFn: () => api.trainingTemplate(templateId!),
    enabled: Boolean(templateId),
  });

  // Apply template defaults the moment the data arrives. Only sft/raft
  // are honored here — other modalities can't be launched from this
  // form yet, so the gallery directs them to the CLI instead.
  useEffect(() => {
    if (!templateQuery.data || templateApplied === templateQuery.data.id) return;
    const t = templateQuery.data;
    if (t.modality !== "sft" && t.modality !== "raft") return;
    const hp = t.hyperparams as Record<string, unknown>;
    setConfig((prev) => ({
      ...prev,
      modality: t.modality as Modality,
      model: t.model_hint || prev.model,
      dataset: t.dataset_hint && t.dataset_hint !== "@custom"
        ? t.dataset_hint
        : prev.dataset,
      verifier: typeof t.verifier === "string" ? t.verifier : prev.verifier,
      epochs: typeof hp.epochs === "number" ? hp.epochs : prev.epochs,
      batchSize: typeof hp.batch_size === "number" ? hp.batch_size : prev.batchSize,
      learningRate: typeof hp.learning_rate === "number"
        ? hp.learning_rate.toString()
        : prev.learningRate,
      loraRank: typeof hp.lora_rank === "number" ? hp.lora_rank : prev.loraRank,
      cycles: typeof hp.cycles === "number" ? hp.cycles : prev.cycles,
      samplesPerPrompt: typeof hp.samples_per_prompt === "number"
        ? hp.samples_per_prompt
        : prev.samplesPerPrompt,
    }));
    setTemplateApplied(t.id);
  }, [templateQuery.data, templateApplied]);

  // Auto-populate the model field from the backend's first suggestion
  // once the suggestion list arrives. Don't overwrite a user-typed value
  // and don't overwrite a template-supplied value.
  useEffect(() => {
    if (modelId || config.model || !models.data?.items.length || templateApplied) return;
    setConfig((prev) => ({ ...prev, model: models.data!.items[0].id }));
  }, [models.data, config.model, templateApplied, modelId]);

  // Live preflight: 400ms debounce on form changes. The mutation status
  // gives us loading / success / error states to render in the side panel.
  useEffect(() => {
    if (!config.model) return;
    const t = window.setTimeout(() => {
      preflight.mutate(buildLaunchPayload(config));
    }, 400);
    return () => window.clearTimeout(t);
    // We intentionally only watch the values that affect preflight, not
    // the entire mutation reference (which would trigger an infinite loop).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    config.modality,
    config.model,
    config.dataset,
    config.customDatasetFile,
    config.verifier,
  ]);

  return (
    <>
      <Topbar
        eyebrow="Workspace"
        title="Training"
        subtitle="Configure and launch RAFT or SFT runs."
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
              onClick={() => setConfig(defaultConfig())}
              type="button"
            >
              Reset
            </Button>
          </>
        }
        statusBar={
          <>
            <ReadoutItem label="MODE" value={config.modality.toUpperCase()} />
            <ReadoutSep />
            <ReadoutItem label="BACKEND" value={backend.data?.name ?? "—"} />
            <ReadoutSep />
            <ReadoutItem
              label="MODEL"
              value={config.model ? truncate(config.model, 28) : "—"}
            />
          </>
        }
      />

      <div className="px-5 py-5 space-y-4">
        <FirstRunPanel
          backendName={backend.data?.name}
          onApply={(next) => setConfig((prev) => ({ ...prev, ...next }))}
        />

        <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-4">
          {/* LEFT: form */}
          <div className="space-y-4">
            <ModalitySection config={config} setConfig={setConfig} />
            <ModelSection
              config={config}
              setConfig={setConfig}
              models={models.data?.items ?? []}
              selectedModel={selectedModel}
            />
            <DatasetSection
              config={config}
              setConfig={setConfig}
              datasets={datasets.data?.items ?? []}
            />
            {config.modality === "raft" ? (
              <VerifierSection
                config={config}
                setConfig={setConfig}
                verifiers={verifiers.data?.items ?? []}
              />
            ) : null}
            <AdvancedSection
              config={config}
              setConfig={setConfig}
              open={advancedOpen}
              onOpenChange={setAdvancedOpen}
            />
          </div>

          {/* RIGHT: preflight + launch (sticky) */}
          <div className="lg:sticky lg:top-4 self-start space-y-3">
            <PreflightPanel
              preflightStatus={currentPreflightStatus}
              checks={buildPreflightChecks(config, preflight, backend.data?.name)}
            />
            <LaunchPanel
              config={config}
              selectedModel={selectedModel}
              disabled={!canLaunch(config) || currentPreflightStatus !== "ok" || launch.isPending}
              launching={launch.isPending}
              onLaunch={() => launch.mutate(buildLaunchPayload(config))}
              launchedRunId={launch.data?.run_id as string | undefined}
              error={(launch.error as Error | null)?.message ?? undefined}
            />
          </div>
        </div>
      </div>
      </>
  );
}

/* -------------------------------------------------------------------------
 * Sections
 * ----------------------------------------------------------------------- */

function FirstRunPanel({
  backendName,
  onApply,
}: {
  backendName?: string;
  onApply: (partial: Partial<ConfigState>) => void;
}) {
  const appleMlx = backendName === "mlx";
  const appleTorch = backendName === "mps";
  const firstModel = appleMlx
    ? "mlx-community/Qwen2.5-3B-Instruct-bf16"
    : "Qwen/Qwen2.5-Coder-1.5B";

  return (
    <div className="rounded-lg border border-border bg-surface px-4 py-3">
      <div className="grid gap-3 lg:grid-cols-[1fr_auto] lg:items-center">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[11px] font-medium uppercase tracking-wider text-fg-subtle">
              First successful run
            </span>
            <span className="rounded-sm border border-border-subtle px-1.5 py-0.5 font-mono text-[10px] text-fg-disabled">
              {backendName ? `backend:${backendName}` : "detecting backend"}
            </span>
          </div>
          <p className="mt-1 text-[13px] text-fg-muted">
            Start with a small catalog model, a known dataset, and conservative defaults.
            Switch to RAFT once SFT and preflight are clean.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button
            type="button"
            size="sm"
            variant="secondary"
            onClick={() =>
              onApply({
                modality: "sft",
                model: firstModel,
                dataset: "codealpaca",
                epochs: 1,
                batchSize: appleMlx || appleTorch ? 1 : 2,
              })
            }
          >
            <CheckCircle2 />
            Safe SFT
          </Button>
          <Button
            type="button"
            size="sm"
            variant="secondary"
            onClick={() =>
              onApply({
                modality: "raft",
                model: "Qwen/Qwen2.5-Coder-1.5B",
                dataset: "codealpaca",
                verifier: "gcc",
                cycles: 2,
                samplesPerPrompt: 4,
              })
            }
          >
            <ClipboardList />
            Code RAFT
          </Button>
          <Button asChild size="sm" variant="ghost">
            <Link to="/models">
              <BookOpen />
              Catalog
            </Link>
          </Button>
        </div>
      </div>
    </div>
  );
}

function ModalitySection({
  config,
  setConfig,
}: {
  config: ConfigState;
  setConfig: SetConfig;
}) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardEyebrow>STEP 01</CardEyebrow>
          <CardTitle>Modality</CardTitle>
        </div>
      </CardHeader>
      <CardContent>
        <RadioCardGroup
          value={config.modality}
          onValueChange={(v) => setConfig((c) => ({ ...c, modality: v as Modality }))}
          className="grid grid-cols-1 md:grid-cols-2 gap-2"
        >
          <RadioCard
            value="sft"
            title="SFT"
            description="Supervised fine-tuning on a labelled dataset. Single pass, deterministic, fastest."
          />
          <RadioCard
            value="raft"
            title="RAFT"
            description="Reward-ranked fine-tuning. Generate, verify, filter, train; iterate across cycles."
          />
        </RadioCardGroup>
      </CardContent>
    </Card>
  );
}

function ModelSection({
  config,
  setConfig,
  models,
  selectedModel,
}: {
  config: ConfigState;
  setConfig: SetConfig;
  models: ModelCatalogEntry[];
  selectedModel: ModelCatalogEntry | null;
}) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardEyebrow>STEP 02</CardEyebrow>
          <CardTitle>Base model</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="space-y-2.5">
        <Label htmlFor="model-id">Model identifier (HuggingFace or MLX-format repo)</Label>
        <Input
          id="model-id"
          mono
          placeholder="Qwen/Qwen2.5-Coder-3B"
          value={config.model}
          onChange={(e) => setConfig((c) => ({ ...c, model: e.target.value }))}
        />
        {models.length > 0 ? (
          <div className="flex flex-wrap gap-1.5 pt-1">
            <span className="text-[11px] uppercase tracking-wider text-fg-disabled font-medium pr-1">
              Suggestions
            </span>
            {models.map((m) => (
              <button
                key={m.id}
                type="button"
                onClick={() => setConfig((c) => ({ ...c, model: m.id }))}
                title={(m.known_caveats ?? []).length ? (m.known_caveats ?? []).join(" ") : m.recommended_use}
                className={cn(
                  "px-2 py-0.5 rounded-sm border text-[11px] font-mono transition-colors",
                  m.id === config.model
                    ? "border-accent bg-accent-bg text-accent"
                    : "border-border-subtle text-fg-muted hover:border-border-strong hover:bg-surface",
                )}
              >
                {m.id}
              </button>
            ))}
          </div>
        ) : null}
        {selectedModel ? <SelectedModelInsight model={selectedModel} /> : null}
      </CardContent>
    </Card>
  );
}

function SelectedModelInsight({ model }: { model: ModelCatalogEntry }) {
  const caveats = [
    ...(model.known_caveats ?? []),
    ...(model.trust_remote_code_required ? ["Requires explicit trust_remote_code opt-in."] : []),
  ];
  const fitNotes = model.fit_notes ?? [];

  return (
    <div className="mt-3 rounded-md border border-border-subtle bg-bg-subtle/50 px-3 py-2.5">
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="font-medium text-[12px] text-fg">{model.label || model.id}</span>
        <span className="font-mono text-[10px] text-fg-disabled">{model.parameter_count || "size unknown"}</span>
        <span className="font-mono text-[10px] text-fg-disabled">{model.memory_tier || "memory unknown"}</span>
        {model.estimated_memory_gb ? (
          <span className="font-mono text-[10px] text-fg-disabled">
            ~{model.estimated_memory_gb}GB
          </span>
        ) : null}
        <span className="font-mono text-[10px] text-fg-disabled">{model.status || "catalog"}</span>
        {model.recommended_first_run ? (
          <span className="font-mono text-[10px] text-success">first-run</span>
        ) : null}
      </div>
      <p className="mt-1 text-[12px] text-fg-muted">
        {model.recommended_use || "Catalog metadata is still loading for this suggestion."}
      </p>
      <div className="mt-2 flex flex-wrap gap-1.5">
        {(model.backend_support ?? []).map((backend) => (
          <span
            key={backend}
            className="rounded-sm border border-border-subtle px-1.5 py-0.5 font-mono text-[10px] text-fg-subtle"
          >
            {backend}
          </span>
        ))}
      </div>
      {model.mlx_variant ? (
        <div className="mt-2 font-mono text-[10.5px] text-fg-subtle">
          MLX variant: {model.mlx_variant}
        </div>
      ) : null}
      {fitNotes.length ? (
        <div className="mt-2 space-y-1 text-[11px] text-fg-subtle">
          {fitNotes.map((note) => (
            <div key={note}>{note}</div>
          ))}
        </div>
      ) : null}
      {model.license_note || model.download_note ? (
        <div className="mt-2 space-y-1 text-[11px] text-warning">
          {model.license_note ? <div>{model.license_note}</div> : null}
          {model.download_note ? <div>{model.download_note}</div> : null}
        </div>
      ) : null}
      {caveats.length ? (
        <div className="mt-2 space-y-1 text-[11px] text-warning">
          {caveats.map((caveat) => (
            <div key={caveat}>{caveat}</div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function DatasetSection({
  config,
  setConfig,
  datasets,
}: {
  config: ConfigState;
  setConfig: SetConfig;
  datasets: { key: string; description: string; size_hint: string; domain: string }[];
}) {
  // Group by domain so the picker doesn't become a flat 30-item list.
  const grouped = useMemo(() => {
    const out: Record<string, typeof datasets> = {};
    for (const d of datasets) {
      (out[d.domain] ||= []).push(d);
    }
    return out;
  }, [datasets]);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardEyebrow>STEP 03</CardEyebrow>
          <CardTitle>Dataset</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="space-y-1.5">
          <Label htmlFor="dataset">Pre-registered datasets</Label>
          <Select
            value={config.dataset}
            onValueChange={(v) => setConfig((c) => ({ ...c, dataset: v }))}
          >
            <SelectTrigger id="dataset">
              <SelectValue placeholder="Select a dataset" />
            </SelectTrigger>
            <SelectContent>
              {Object.entries(grouped).map(([domain, items]) => (
                <div key={domain}>
                  <div className="px-2 py-1 text-[10px] uppercase tracking-wider text-fg-disabled font-medium">
                    {domain}
                  </div>
                  {items.map((d) => (
                    <SelectItem key={d.key} value={d.key}>
                      <span className="flex items-center gap-2">
                        <span className="font-mono text-[12px]">{d.key}</span>
                        <span className="text-fg-subtle text-[11px]">· {d.size_hint}</span>
                      </span>
                    </SelectItem>
                  ))}
                </div>
              ))}
              <SelectItem value="__custom__">Custom JSONL file…</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {config.dataset === "__custom__" ? (
          <div className="space-y-1.5">
            <Label htmlFor="dataset-file">JSONL file path</Label>
            <Input
              id="dataset-file"
              mono
              placeholder="/path/to/training.jsonl"
              value={config.customDatasetFile}
              onChange={(e) =>
                setConfig((c) => ({ ...c, customDatasetFile: e.target.value }))
              }
            />
            <p className="text-[11px] text-fg-subtle">
              Each line should be a JSON object with `text` or `messages`. The MLX
              path also accepts `prompt` + `completion`.
            </p>
          </div>
        ) : (
          <DatasetSummary datasets={datasets} selected={config.dataset} />
        )}
      </CardContent>
    </Card>
  );
}

function DatasetSummary({
  datasets,
  selected,
}: {
  datasets: { key: string; description: string; size_hint: string; huggingface_id?: string }[];
  selected: string;
}) {
  const ds = datasets.find((d) => d.key === selected);
  if (!ds) return null;
  return (
    <div className="rounded-md border border-border-subtle bg-bg-subtle/50 px-3 py-2 text-[12px]">
      <div className="text-fg">{ds.description}</div>
      {ds.huggingface_id ? (
        <div className="font-mono text-[11px] text-fg-subtle mt-0.5">
          {ds.huggingface_id} · {ds.size_hint}
        </div>
      ) : null}
    </div>
  );
}

function VerifierSection({
  config,
  setConfig,
  verifiers,
}: {
  config: ConfigState;
  setConfig: SetConfig;
  verifiers: { key: string; label: string; toolchain: string }[];
}) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardEyebrow>STEP 04</CardEyebrow>
          <CardTitle>Verifier</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="space-y-1.5">
        <Label htmlFor="verifier">RAFT verifier toolchain</Label>
        <Select
          value={config.verifier}
          onValueChange={(v) => setConfig((c) => ({ ...c, verifier: v }))}
        >
          <SelectTrigger id="verifier">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {verifiers.map((v) => (
              <SelectItem key={v.key} value={v.key}>
                <span className="flex items-center gap-2">
                  <span className="font-mono text-[12px]">{v.key}</span>
                  <span className="text-fg-subtle text-[11px]">· {v.label}</span>
                </span>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <p className="text-[11px] text-fg-subtle pt-1">
          The verifier scores generated samples; only those passing the reward
          threshold flow into the SFT-on-accepted-samples step.
        </p>
      </CardContent>
    </Card>
  );
}

function AdvancedSection({
  config,
  setConfig,
  open,
  onOpenChange,
}: {
  config: ConfigState;
  setConfig: SetConfig;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  return (
    <Card>
      <Collapsible open={open} onOpenChange={onOpenChange}>
        <CollapsibleTrigger asChild>
          <button
            type="button"
            className="flex w-full items-center justify-between border-b border-border-subtle px-4 py-3 hover:bg-surface-hover/30 transition-colors"
          >
            <div className="flex items-center gap-2">
              <CardEyebrow>OPTIONAL</CardEyebrow>
              <CardTitle>Hyperparameters</CardTitle>
              <Settings2 className="h-3.5 w-3.5 text-fg-subtle ml-1" />
            </div>
            <ChevronRight
              className={cn(
                "h-4 w-4 text-fg-subtle transition-transform",
                open && "rotate-90",
              )}
            />
          </button>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <CardContent className="grid grid-cols-2 md:grid-cols-3 gap-x-4 gap-y-3">
            <Field
              label="Epochs"
              value={config.epochs}
              onChange={(v) => setConfig((c) => ({ ...c, epochs: v as number }))}
              type="number"
            />
            <Field
              label="Batch size"
              value={config.batchSize}
              onChange={(v) => setConfig((c) => ({ ...c, batchSize: v as number }))}
              type="number"
            />
            <Field
              label="Learning rate"
              value={config.learningRate}
              onChange={(v) =>
                setConfig((c) => ({ ...c, learningRate: String(v) }))
              }
              type="text"
              hint="e.g. 2e-4"
            />
            <Field
              label="LoRA rank"
              value={config.loraRank}
              onChange={(v) => setConfig((c) => ({ ...c, loraRank: v as number }))}
              type="number"
            />
            <Field
              label="LoRA alpha"
              value={config.loraAlpha}
              onChange={(v) => setConfig((c) => ({ ...c, loraAlpha: v as number }))}
              type="number"
            />
            <Field
              label="Max seq length"
              value={config.maxSeqLength}
              onChange={(v) => setConfig((c) => ({ ...c, maxSeqLength: v as number }))}
              type="number"
            />
            {config.modality === "raft" ? (
              <>
                <Field
                  label="Cycles"
                  value={config.cycles}
                  onChange={(v) => setConfig((c) => ({ ...c, cycles: v as number }))}
                  type="number"
                />
                <Field
                  label="Samples / prompt"
                  value={config.samplesPerPrompt}
                  onChange={(v) =>
                    setConfig((c) => ({ ...c, samplesPerPrompt: v as number }))
                  }
                  type="number"
                />
              </>
            ) : null}
          </CardContent>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  );
}

function Field({
  label,
  value,
  onChange,
  type,
  hint,
}: {
  label: string;
  value: string | number;
  onChange: (v: string | number) => void;
  type: "text" | "number";
  hint?: string;
}) {
  return (
    <div className="space-y-1.5">
      <Label>{label}</Label>
      <Input
        mono
        type={type}
        value={value}
        onChange={(e) =>
          onChange(type === "number" ? Number(e.target.value) || 0 : e.target.value)
        }
      />
      {hint ? (
        <span className="font-mono text-[10px] text-fg-disabled">{hint}</span>
      ) : null}
    </div>
  );
}

/* -------------------------------------------------------------------------
 * Preflight + launch panels
 * ----------------------------------------------------------------------- */

interface PreflightCheck {
  label: string;
  status: "ok" | "warning" | "error" | "loading" | "pending";
  detail?: string;
}

function PreflightPanel({
  checks,
  preflightStatus,
}: {
  checks: PreflightCheck[];
  preflightStatus: "idle" | "loading" | "ok" | "error";
}) {
  const overallTone =
    preflightStatus === "loading"
      ? "loading"
      : checks.some((c) => c.status === "error")
        ? "error"
        : checks.some((c) => c.status === "warning")
          ? "warning"
          : preflightStatus === "ok"
            ? "ok"
            : "pending";

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardEyebrow>STATUS</CardEyebrow>
          <CardTitle>Preflight</CardTitle>
        </div>
        <PreflightOverall tone={overallTone} />
      </CardHeader>
      <CardContent className="p-0 divide-y divide-border-subtle">
        {checks.map((c) => (
          <div key={c.label} className="flex items-center gap-2.5 px-4 py-2">
            <CheckIcon status={c.status} />
            <div className="min-w-0 flex-1">
              <div className="text-[12.5px] text-fg leading-tight">{c.label}</div>
              {c.detail ? (
                <div className="text-[11px] text-fg-subtle truncate">{c.detail}</div>
              ) : null}
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

function PreflightOverall({
  tone,
}: {
  tone: "ok" | "loading" | "error" | "warning" | "pending";
}) {
  const map = {
    ok: { Icon: CheckCircle2, color: "text-success", label: "Ready" },
    loading: { Icon: Loader2, color: "text-fg-subtle animate-spin", label: "Checking" },
    error: { Icon: XCircle, color: "text-danger", label: "Issues" },
    warning: { Icon: AlertTriangle, color: "text-warning", label: "Caution" },
    pending: { Icon: CircleDashed, color: "text-fg-disabled", label: "Idle" },
  } as const;
  const { Icon, color, label } = map[tone];
  return (
    <div className={cn("flex items-center gap-1.5 text-[11px] font-medium", color)}>
      <Icon className="h-3.5 w-3.5" />
      {label}
    </div>
  );
}

function CheckIcon({ status }: { status: PreflightCheck["status"] }) {
  switch (status) {
    case "ok":
      return <CheckCircle2 className="h-3.5 w-3.5 text-success shrink-0" />;
    case "warning":
      return <AlertTriangle className="h-3.5 w-3.5 text-warning shrink-0" />;
    case "error":
      return <XCircle className="h-3.5 w-3.5 text-danger shrink-0" />;
    case "loading":
      return <Loader2 className="h-3.5 w-3.5 text-fg-subtle shrink-0 animate-spin" />;
    case "pending":
    default:
      return <CircleDashed className="h-3.5 w-3.5 text-fg-disabled shrink-0" />;
  }
}

function LaunchPanel({
  config,
  selectedModel,
  disabled,
  launching,
  onLaunch,
  launchedRunId,
  error,
}: {
  config: ConfigState;
  selectedModel: ModelCatalogEntry | null;
  disabled: boolean;
  launching: boolean;
  onLaunch: () => void;
  launchedRunId?: string;
  error?: string;
}) {
  return (
    <Card>
      <CardContent className="space-y-2.5 p-4">
        <Button
          variant="primary"
          size="lg"
          className="w-full"
          disabled={disabled}
          onClick={onLaunch}
        >
          {launching ? (
            <>
              <Loader2 className="animate-spin" />
              Launching…
            </>
          ) : (
            <>
              <Play />
              Launch run
            </>
          )}
        </Button>
        {launchedRunId ? (
          <div className="flex items-center gap-2 rounded-md border border-success/30 bg-success-bg px-3 py-2">
            <Sparkles className="h-3.5 w-3.5 text-success" />
            <span className="font-mono text-[11px] text-success">
              Started {launchedRunId}
            </span>
          </div>
        ) : null}
        {error ? (
          <div className="rounded-md border border-danger/30 bg-danger-bg px-3 py-2 text-[11.5px] text-danger">
            {error}
          </div>
        ) : null}
        <div className="rounded-md border border-border-subtle bg-bg-subtle/50 px-3 py-2 text-[11px] text-fg-subtle leading-relaxed">
          <FileQuestion className="h-3 w-3 inline mr-1 -mt-0.5 text-fg-disabled" />
          Cost + duration estimates land in phase D once live runs are wired.
        </div>
        <RunSummary config={config} selectedModel={selectedModel} />
      </CardContent>
    </Card>
  );
}

function RunSummary({
  config,
  selectedModel,
}: {
  config: ConfigState;
  selectedModel: ModelCatalogEntry | null;
}) {
  const rows = [
    ["Mode", config.modality.toUpperCase()],
    ["Model", config.model || "not set"],
    ["Dataset", config.dataset === "__custom__" ? config.customDatasetFile || "custom path needed" : config.dataset],
    ["Memory", selectedModel?.estimated_memory_gb ? `${selectedModel.memory_tier || "unknown"} · ~${selectedModel.estimated_memory_gb}GB` : selectedModel?.memory_tier ?? "unknown"],
    ["Output", `models/${config.modality}-${slug(config.model || "model")}`],
  ];

  if (config.modality === "raft") {
    rows.splice(
      3,
      0,
      ["Verifier", config.verifier],
      ["Cycles", String(config.cycles)],
      ["Samples", `${config.samplesPerPrompt}/prompt`],
    );
  }

  return (
    <div className="rounded-md border border-border-subtle bg-bg-subtle/50">
      <div className="border-b border-border-subtle px-3 py-2 text-[10px] font-medium uppercase tracking-wider text-fg-disabled">
        Launch summary
      </div>
      <dl className="divide-y divide-border-subtle">
        {rows.map(([label, value]) => (
          <div key={label} className="grid grid-cols-[72px_1fr] gap-2 px-3 py-1.5">
            <dt className="text-[10.5px] uppercase tracking-wider text-fg-disabled">{label}</dt>
            <dd className="truncate font-mono text-[11px] text-fg-subtle" title={value}>
              {value}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

/* -------------------------------------------------------------------------
 * Helpers
 * ----------------------------------------------------------------------- */

type SetConfig = (updater: (c: ConfigState) => ConfigState) => void;

function buildLaunchPayload(c: ConfigState): Record<string, unknown> {
  // The public API enforces a tight allow-list of fields per modality
  // (PUBLIC_TRAIN_ALLOWED_FIELDS in halo_forge/public_api/service.py).
  // Any unsupported key triggers a 400. We only forward fields the
  // backend accepts; the rest of the configurator state lives on the
  // client until phase E broadens the API.
  const lr = parseFloat(c.learningRate);
  const isCustom = c.dataset === "__custom__";

  if (c.modality === "sft") {
    return stripEmpty({
      mode: "sft",
      model: c.model,
      // SFT: registered dataset key OR custom path. The training service
      // accepts either string in `dataset`; the path is detected by
      // suffix (.jsonl) and routed to load_local_dataset internally.
      dataset: isCustom ? c.customDatasetFile : c.dataset,
      output_dir: `models/sft-${slug(c.model)}`,
      epochs: c.epochs,
      batch_size: c.batchSize,
      learning_rate: Number.isFinite(lr) ? lr : undefined,
    });
  }

  // RAFT: the backend takes `prompts` (a file path), not `dataset`. We
  // surface this in the UI as the same dataset/file picker; registered
  // datasets are forwarded as their HF id (the launch service knows how
  // to convert), custom files are forwarded directly.
  return stripEmpty({
    mode: "raft",
    model: c.model,
    prompts: isCustom ? c.customDatasetFile : c.dataset,
    output_dir: `models/raft-${slug(c.model)}`,
    cycles: c.cycles,
    samples_per_prompt: c.samplesPerPrompt,
    temperature: undefined, // wired in advanced section in a future iteration
  });
}

function stripEmpty(o: Record<string, unknown>): Record<string, unknown> {
  // Drop keys whose value is undefined / "" — keeps the payload tight
  // and matches the backend's `_has_public_value` filter.
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(o)) {
    if (v === undefined || v === null || v === "") continue;
    out[k] = v;
  }
  return out;
}

function slug(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 32);
}

function canLaunch(c: ConfigState): boolean {
  if (!c.model.trim()) return false;
  if (c.dataset === "__custom__" && !c.customDatasetFile.trim()) return false;
  if (!c.dataset) return false;
  return true;
}

function preflightStatus(
  preflight: ReturnType<typeof useTrainingPreflight>,
  config: ConfigState,
): "idle" | "loading" | "ok" | "error" {
  if (!config.model) return "idle";
  if (preflight.isPending) return "loading";
  if (preflight.isError) return "error";
  if (preflight.isSuccess) return preflight.data.ok ? "ok" : "error";
  return "idle";
}

function buildPreflightChecks(
  config: ConfigState,
  preflight: ReturnType<typeof useTrainingPreflight>,
  backendName: string | undefined,
): PreflightCheck[] {
  const checks: PreflightCheck[] = [
    {
      label: "Backend connected",
      status: backendName ? "ok" : "loading",
      detail: backendName ? `Active accelerator: ${backendName}` : "Detecting…",
    },
    {
      label: "Model identifier set",
      status: config.model ? "ok" : "pending",
      detail: config.model || "Type a HuggingFace or MLX repo id above",
    },
    {
      label: "Dataset selected",
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

  if (config.modality === "raft") {
    checks.push({
      label: "Verifier toolchain",
      status: config.verifier ? "ok" : "pending",
      detail: config.verifier
        ? `${config.verifier} (preflight will check binary on the host)`
        : "Pick a verifier above",
    });
  }

  if (preflight.isPending) {
    checks.push({ label: "Server preflight", status: "loading", detail: "Validating launch…" });
  } else if (preflight.isError) {
    const msg = (preflight.error as Error | null)?.message ?? "Preflight failed";
    checks.push({ label: "Server preflight", status: "error", detail: msg });
  } else if (preflight.isSuccess && preflight.data) {
    const data = preflight.data;
    const issue = data.errors[0] ?? data.suggested_fixes[0];
    const summary =
      issue ??
      data.user_summary?.headline ??
      (data.ok ? "All checks passed" : "Preflight found launch issues");
    checks.push({
      label: "Server preflight",
      status: data.ok ? (data.warnings.length ? "warning" : "ok") : "error",
      detail: summary,
    });
  } else {
    checks.push({
      label: "Server preflight",
      status: "pending",
      detail: "Runs automatically as you edit the form",
    });
  }

  return checks;
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
  return <span className="text-fg-disabled select-none">·</span>;
}

function truncate(s: string, n: number): string {
  return s.length > n ? `${s.slice(0, n - 1)}…` : s;
}
