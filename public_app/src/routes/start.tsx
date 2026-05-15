import { createFileRoute, Link } from "@tanstack/react-router";
import {
  ArrowRight,
  AlertTriangle,
  Bot,
  CheckCircle2,
  CircleDashed,
  Code2,
  Cpu,
  Loader2,
  Play,
  Rocket,
  Sigma,
  XCircle,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Topbar } from "@/components/shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  useBackendInfo,
  useModelCatalog,
  useTrainingDatasets,
  useTrainingLaunch,
  useTrainingModels,
  useTrainingPreflight,
  useWorkspaceInfo,
} from "@/lib/hooks";
import type { BackendInfo, ModelCatalogEntry } from "@/lib/api";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/start")({
  validateSearch: (search: Record<string, unknown>) => ({
    goal: typeof search.goal === "string" ? search.goal : undefined,
  }),
  component: StartRoute,
});

type StartGoalId = "code" | "reasoning" | "tool-use" | "apple-silicon";

type StartGoal = {
  id: StartGoalId;
  title: string;
  body: string;
  mode: "sft";
  dataset: string;
  query: string;
  modelFallback: string;
  outputPrefix: string;
  icon: typeof Code2;
  requiresApple?: boolean;
};

const START_GOALS: StartGoal[] = [
  {
    id: "code",
    title: "Code",
    body: "Validate local fine-tuning with a small coder model and CodeAlpaca.",
    mode: "sft",
    dataset: "codealpaca",
    query: "code",
    modelFallback: "Qwen/Qwen2.5-Coder-0.5B",
    outputPrefix: "code",
    icon: Code2,
  },
  {
    id: "reasoning",
    title: "Reasoning",
    body: "Run a small math/reasoning SFT smoke test with GSM8K-format data.",
    mode: "sft",
    dataset: "gsm8k_sft",
    query: "reasoning",
    modelFallback: "Qwen/Qwen2.5-1.5B-Instruct",
    outputPrefix: "reasoning",
    icon: Sigma,
  },
  {
    id: "tool-use",
    title: "Tool use",
    body: "Check function-calling data flow with a compact agentic SFT run.",
    mode: "sft",
    dataset: "xlam_sft",
    query: "agentic",
    modelFallback: "Qwen/Qwen2.5-1.5B-Instruct",
    outputPrefix: "tool-use",
    icon: Bot,
  },
  {
    id: "apple-silicon",
    title: "Apple Silicon",
    body: "Use MLX-friendly code defaults when this workstation is ready for MLX.",
    mode: "sft",
    dataset: "codealpaca",
    query: "mlx code",
    modelFallback: "mlx-community/Qwen2.5-0.5B-Instruct-bf16",
    outputPrefix: "mlx-code",
    icon: Cpu,
    requiresApple: true,
  },
];

function StartRoute() {
  const search = Route.useSearch();
  const backend = useBackendInfo();
  const workspace = useWorkspaceInfo();
  const datasets = useTrainingDatasets();
  const models = useTrainingModels({ mode: "sft" });
  const mlxModels = useModelCatalog({ mode: "sft", backend: "mlx" });
  const catalog = useModelCatalog({ mode: "sft" });
  const preflight = useTrainingPreflight();
  const launch = useTrainingLaunch();
  const lastPreflightKey = useRef<string | null>(null);
  const initialGoal = START_GOALS.find((goal) => goal.id === search.goal)?.id ?? "code";
  const [goalId, setGoalId] = useState<StartGoalId>(initialGoal);

  const mlxReadiness = backend.data?.mlx_readiness;
  const mlxReady = mlxReadiness?.executable === true;
  const appleSilicon = Boolean(
    backend.data?.chip ||
      backend.data?.name === "mps" ||
      backend.data?.name === "mlx" ||
      mlxReadiness?.macos_version ||
      mlxReadiness?.metal_device,
  );
  const selectedGoal = START_GOALS.find((goal) => goal.id === goalId) ?? START_GOALS[0];
  const modelCandidates = useMemo(() => {
    const raw =
      selectedGoal.id === "apple-silicon" && mlxReady
        ? (mlxModels.data?.items ?? [])
        : (catalog.data?.items ?? models.data?.items ?? []);
    const backendName = backend.data?.name;
    if (!backendName) return raw;
    const compatible = raw.filter((item) => item.backend_support.includes(backendName));
    return compatible.length ? compatible : raw;
  }, [
    backend.data?.name,
    catalog.data?.items,
    mlxModels.data?.items,
    mlxReady,
    models.data?.items,
    selectedGoal.id,
  ]);
  const recommended = useMemo(
    () => selectGoalModel(modelCandidates, selectedGoal),
    [modelCandidates, selectedGoal],
  );
  const dataset =
    datasets.data?.items.find((item) => item.key === selectedGoal.dataset)?.key ??
    selectedGoal.dataset;
  const payload = useMemo(
    () =>
      backend.data && workspace.data && recommended && dataset
        ? {
            mode: selectedGoal.mode,
            model: recommended.id,
            dataset,
            output_dir: joinPath(
              workspace.data.default_run_root,
              `start-${selectedGoal.outputPrefix}-${slug(recommended.id)}`,
            ),
            accelerator: selectedGoal.id === "apple-silicon" && mlxReady ? "mlx" : undefined,
            epochs: 1,
            batch_size:
              selectedGoal.id === "apple-silicon" ||
              mlxReady ||
              backend.data?.name === "mlx" ||
              backend.data?.name === "mps"
                ? 1
                : 2,
            learning_rate: 2e-4,
            max_samples: 200,
        }
        : null,
    [backend.data, dataset, mlxReady, recommended, selectedGoal, workspace.data],
  );

  const payloadKey = payload ? JSON.stringify(payload) : "";

  useEffect(() => {
    if (backend.data && selectedGoal.requiresApple && !appleSilicon) {
      setGoalId("code");
    }
  }, [appleSilicon, backend.data, selectedGoal.requiresApple]);

  useEffect(() => {
    if (!payload || preflight.isPending || lastPreflightKey.current === payloadKey) return;
    lastPreflightKey.current = payloadKey;
    preflight.mutate(payload);
  }, [payload, payloadKey, preflight]);

  const runId = extractRunId(launch.data);
  const preflightOk = preflight.isSuccess && preflight.data?.ok === true;
  const preflightBlocked = preflight.isSuccess && preflight.data?.ok === false;
  const ready = Boolean(payload && preflightOk && !preflight.isError);
  const launchLabel = launch.isPending
    ? "Launching..."
    : preflightBlocked
      ? "Fix preflight"
      : preflight.isPending
        ? "Checking..."
        : `Launch ${selectedGoal.title.toLowerCase()} run`;
  const workspaceBlocked = Boolean(workspace.data && !workspace.data.writable);
  const preflightBody = workspace.isLoading
    ? "Finding writable run folder"
    : workspaceBlocked
      ? workspace.data?.message ?? "Default run folder is not writable"
      : preflight.isError
        ? (preflight.error as Error).message
        : preflight.isSuccess
          ? preflight.data.ok
            ? "Launch checks passed"
            : firstPreflightIssue(preflight.data) ?? "Preflight found launch issues"
          : "Waiting for generated launch payload";
  const preflightTone = workspaceBlocked || preflight.isError || preflightBlocked
    ? "danger"
    : ready
      ? "success"
      : "neutral";

  return (
    <>
      <Topbar
        eyebrow="Workspace"
        title="Start"
        subtitle="Pick a goal. Halo Forge chooses safe SFT defaults and checks the launch first."
        actions={
          <Button asChild variant="ghost" size="sm">
            <Link to="/train">
              Advanced training <ArrowRight />
            </Link>
          </Button>
        }
      />

      <div className="px-5 py-5 space-y-4">
        <section className="grid gap-2 lg:grid-cols-4">
          {START_GOALS.map((goal) => (
            <GoalButton
              key={goal.id}
              goal={goal}
              active={goal.id === selectedGoal.id}
              disabled={goal.requiresApple && !appleSilicon}
              onClick={() => {
                setGoalId(goal.id);
                lastPreflightKey.current = null;
                preflight.reset();
                launch.reset();
              }}
            />
          ))}
        </section>

        <div className="rounded-lg border border-border bg-surface px-4 py-3">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
            <div className="min-w-0 max-w-[86ch]">
              <div className="flex items-center gap-2">
                <Rocket className="h-4 w-4 text-accent" />
                <span className="text-sm font-semibold text-fg">{selectedGoal.title} first run</span>
              </div>
              <p className="mt-1 text-[13px] text-fg-muted">
                {selectedGoal.body} The page keeps the settings conservative and lets preflight
                catch missing data, model, or backend issues before launch.
              </p>
            </div>
            <Button
              variant="primary"
              size="md"
              className="shrink-0"
              disabled={!ready || launch.isPending}
              onClick={() => payload && launch.mutate(payload)}
            >
              {launch.isPending ? <Loader2 className="animate-spin" /> : <Play />}
              {launchLabel}
            </Button>
          </div>
        </div>

        <div className={`grid gap-3 ${appleSilicon ? "lg:grid-cols-4" : "lg:grid-cols-3"}`}>
          <StepCard
            step="01"
            title="Backend"
            ready={Boolean(backend.data)}
            loading={backend.isLoading}
            body={backend.data ? `${backend.data.name} · ${backend.data.device}` : "Detecting accelerator"}
          />
          {appleSilicon ? (
            <StepCard
              step="02"
              title="MLX Ready"
              ready={mlxReady}
              loading={backend.isLoading}
              body={mlxReady ? "MLX executable probe passed" : mlxReadinessLabel(mlxReadiness)}
              detail={mlxPackageLabel(mlxReadiness)}
              tone={mlxReady ? "success" : mlxReadiness ? "danger" : "neutral"}
            />
          ) : null}
          <StepCard
            step={appleSilicon ? "03" : "02"}
            title="Model"
            ready={Boolean(recommended)}
            loading={mlxReady ? mlxModels.isLoading : models.isLoading}
            body={recommended ? recommended.id : "Selecting a first-run catalog model"}
            detail={
              recommended
                ? `${modelMemoryLabel(recommended)} · ${modelRiskLabel(recommended)}`
                : undefined
            }
          />
          <StepCard
            step={appleSilicon ? "04" : "03"}
            title="Preflight"
            ready={ready}
            loading={preflight.isPending || workspace.isLoading}
            body={preflightBody}
            tone={preflightTone}
          />
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Generated launch</CardTitle>
            {recommended?.recommended_first_run ? (
              <Badge tone="success" size="sm">best first pick</Badge>
            ) : null}
          </CardHeader>
          <CardContent className="space-y-3">
            {payload ? <LaunchPreview payload={payload} model={recommended} /> : null}
            {preflight.data ? <PreflightMessages data={preflight.data} /> : null}
            {runId ? (
              <div className="rounded-md border border-success/30 bg-success-bg px-3 py-3">
                <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                  <div>
                    <div className="text-[12px] font-medium text-success">Run started</div>
                    <div className="mt-1 font-mono text-[11px] text-success">{runId}</div>
                    {payload ? (
                      <div className="mt-1 text-[11px] text-fg-muted">
                        {String(payload.model)} · {String(payload.dataset)}
                      </div>
                    ) : null}
                    {payload?.output_dir ? (
                      <div className="mt-1 text-[11px] text-fg-muted">
                        Local output path: <span className="font-mono text-fg-subtle">{String(payload.output_dir)}</span>
                      </div>
                    ) : null}
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <Button asChild size="sm" variant="primary">
                      <Link to="/runs/$runId" params={{ runId }}>
                        Open run <ArrowRight />
                      </Link>
                    </Button>
                    <Button asChild size="sm" variant="ghost">
                      <Link to="/runs">
                        View runs
                      </Link>
                    </Button>
                    <Button asChild size="sm" variant="ghost">
                      <Link to="/results">
                        Serve when complete
                      </Link>
                    </Button>
                  </div>
                </div>
              </div>
            ) : null}
            {launch.error ? (
              <div className="rounded-md border border-danger/30 bg-danger-bg px-3 py-2 text-[12px] text-danger">
                {(launch.error as Error).message}
              </div>
            ) : null}
          </CardContent>
        </Card>
      </div>
    </>
  );
}

function GoalButton({
  goal,
  active,
  disabled,
  onClick,
}: {
  goal: StartGoal;
  active: boolean;
  disabled?: boolean;
  onClick: () => void;
}) {
  const Icon = goal.icon;
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className={cn(
        "rounded-lg border px-3 py-3 text-left transition-colors",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent",
        active
          ? "border-accent bg-accent/10"
          : "border-border bg-surface hover:border-border-strong hover:bg-surface-hover/40",
        disabled && "cursor-not-allowed opacity-50 hover:border-border hover:bg-surface",
      )}
      aria-pressed={active}
    >
      <div className="flex items-center gap-2">
        <Icon className={cn("h-4 w-4", active ? "text-accent" : "text-fg-subtle")} />
        <span className="text-[13px] font-semibold text-fg">{goal.title}</span>
        {active ? (
          <Badge tone="success" size="sm">
            selected
          </Badge>
        ) : null}
      </div>
      <p className="mt-1.5 text-[11.5px] leading-5 text-fg-muted">{goal.body}</p>
      {disabled ? (
        <p className="mt-1 text-[10.5px] text-warning">Apple/MLX not detected on this host.</p>
      ) : null}
    </button>
  );
}

function PreflightMessages({
  data,
}: {
  data: NonNullable<ReturnType<typeof useTrainingPreflight>["data"]>;
}) {
  const messages = data.ok
    ? [
        ...data.suggested_fixes.map((message) => ({ tone: "warning" as const, message })),
        ...data.warnings.map((message) => ({ tone: "warning" as const, message })),
      ]
    : [
        ...data.errors.map((message) => ({ tone: "danger" as const, message })),
        ...data.suggested_fixes.map((message) => ({ tone: "warning" as const, message })),
      ];
  if (!messages.length) return null;

  return (
    <div className="space-y-1.5">
      {messages.slice(0, 3).map(({ tone, message }) => {
        const Icon = tone === "danger" ? XCircle : AlertTriangle;
        return (
          <div
            key={`${tone}-${message}`}
            className={
              tone === "danger"
                ? "flex items-start gap-2 rounded-md border border-danger/30 bg-danger-bg px-3 py-2 text-[11.5px] text-danger"
                : "flex items-start gap-2 rounded-md border border-warning/30 bg-warning-bg px-3 py-2 text-[11.5px] text-warning"
            }
          >
            <Icon className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span>{message}</span>
          </div>
        );
      })}
    </div>
  );
}

function StepCard({
  step,
  title,
  body,
  detail,
  ready,
  loading,
  tone = "neutral",
}: {
  step: string;
  title: string;
  body: string;
  detail?: string;
  ready: boolean;
  loading?: boolean;
  tone?: "success" | "danger" | "neutral";
}) {
  const Icon = loading ? Loader2 : ready ? CheckCircle2 : CircleDashed;
  return (
    <Card>
      <CardContent className="space-y-2 p-4">
        <div className="flex items-center justify-between">
          <span className="font-mono text-[10px] text-fg-disabled">{step}</span>
          <Icon
            className={
              loading
                ? "h-4 w-4 animate-spin text-fg-subtle"
                : tone === "success"
                  ? "h-4 w-4 text-success"
                  : tone === "danger"
                    ? "h-4 w-4 text-danger"
                    : "h-4 w-4 text-fg-disabled"
            }
          />
        </div>
        <div>
          <div className="text-sm font-semibold text-fg">{title}</div>
          <div className="mt-1 truncate text-[12px] text-fg-muted" title={body}>
            {body}
          </div>
          {detail ? <div className="mt-1 font-mono text-[10px] text-fg-disabled">{detail}</div> : null}
        </div>
      </CardContent>
    </Card>
  );
}

function LaunchPreview({
  payload,
  model,
}: {
  payload: Record<string, unknown>;
  model: ModelCatalogEntry | null;
}) {
  const rows = [
    ["Mode", String(payload.mode)],
    payload.accelerator ? ["Accelerator", String(payload.accelerator)] : null,
    ["Model", String(payload.model)],
    ["Dataset", String(payload.dataset)],
    ["Memory", modelMemoryLabel(model)],
    ["Output", String(payload.output_dir)],
  ].filter(Boolean) as string[][];
  return (
    <dl className="rounded-md border border-border-subtle bg-bg-subtle/50">
      {rows.map(([label, value]) => (
        <div key={label} className="grid grid-cols-[88px_1fr] gap-3 border-b border-border-subtle px-3 py-2 last:border-0">
          <dt className="text-[10.5px] uppercase tracking-wider text-fg-disabled">{label}</dt>
          <dd className="truncate font-mono text-[11px] text-fg-subtle" title={value}>
            {value}
          </dd>
        </div>
      ))}
    </dl>
  );
}

function selectGoalModel(items: ModelCatalogEntry[], goal: StartGoal): ModelCatalogEntry | null {
  const q = goal.query.toLowerCase().split(/\s+/).filter(Boolean);
  const matching = items.filter((item) => {
    const haystack = [
      item.id,
      item.label,
      item.family,
      item.recommended_use,
      ...(item.tasks ?? []),
      ...(item.modalities ?? []),
      ...(item.trainer_support ?? []),
      ...(item.backend_support ?? []),
    ]
      .join(" ")
      .toLowerCase();
    return q.every((token) => haystack.includes(token));
  });
  const candidates = matching.length ? matching : items;
  return (
    candidates.find((item) => item.id === goal.modelFallback) ??
    candidates.find((item) => item.recommended_first_run && item.risk_level === "safe") ??
    candidates[0] ??
    null
  );
}

function modelMemoryLabel(model: ModelCatalogEntry | null): string {
  if (!model) return "unknown";
  const tier = model.memory_tier || "unknown tier";
  return model.estimated_memory_gb ? `${tier} · ~${model.estimated_memory_gb}GB` : tier;
}

function modelRiskLabel(model: ModelCatalogEntry | null): string {
  return model?.risk_level || model?.status || "catalog metadata pending";
}

function firstPreflightIssue(
  data: NonNullable<ReturnType<typeof useTrainingPreflight>["data"]>,
): string | null {
  return data.errors[0] ?? data.suggested_fixes[0] ?? data.user_summary?.next_step ?? null;
}

function mlxReadinessLabel(readiness: BackendInfo["mlx_readiness"] | undefined): string {
  if (!readiness) return "Checking MLX runtime";
  const first = readiness.errors[0] ?? readiness.warnings[0];
  return first || `MLX ${readiness.status}`;
}

function mlxPackageLabel(readiness: BackendInfo["mlx_readiness"] | undefined): string | undefined {
  if (!readiness) return undefined;
  const versions = readiness.package_versions ?? {};
  return `mlx ${versions.mlx ?? "missing"} · mlx-lm ${versions["mlx-lm"] ?? "missing"}`;
}

function extractRunId(data: unknown): string | null {
  if (!data || typeof data !== "object") return null;
  const record = data as Record<string, unknown>;
  return String(record.run_id ?? record.id ?? "").trim() || null;
}

function slug(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 32);
}

function joinPath(root: string, child: string): string {
  return `${root.replace(/\/+$/, "")}/${child.replace(/^\/+/, "")}`;
}
