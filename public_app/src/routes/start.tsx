import { createFileRoute, Link } from "@tanstack/react-router";
import {
  ArrowRight,
  CheckCircle2,
  CircleDashed,
  Loader2,
  Play,
  Rocket,
} from "lucide-react";
import { useEffect, useMemo } from "react";
import { Topbar } from "@/components/shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  useBackendInfo,
  useTrainingDatasets,
  useTrainingLaunch,
  useTrainingModels,
  useTrainingPreflight,
} from "@/lib/hooks";
import type { ModelCatalogEntry } from "@/lib/api";

export const Route = createFileRoute("/start")({
  component: StartRoute,
});

function StartRoute() {
  const backend = useBackendInfo();
  const datasets = useTrainingDatasets();
  const models = useTrainingModels({ mode: "sft" });
  const preflight = useTrainingPreflight();
  const launch = useTrainingLaunch();

  const recommended = useMemo(() => selectFirstRunModel(models.data?.items ?? []), [models.data]);
  const dataset = datasets.data?.items.find((item) => item.key === "codealpaca")
    ? "codealpaca"
    : datasets.data?.items[0]?.key ?? "";
  const payload = useMemo(
    () =>
      backend.data && recommended && dataset
        ? {
            mode: "sft",
            model: recommended.id,
            dataset,
            output_dir: `models/first-run-${slug(recommended.id)}`,
            epochs: 1,
            batch_size: backend.data?.name === "mlx" || backend.data?.name === "mps" ? 1 : 2,
            learning_rate: 2e-4,
            max_samples: 200,
          }
        : null,
    [backend.data?.name, dataset, recommended],
  );

  useEffect(() => {
    if (!payload || preflight.isPending || preflight.isSuccess) return;
    preflight.mutate(payload);
    // Preflight should fire once for this generated payload.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [payload]);

  const runId = extractRunId(launch.data);
  const ready = Boolean(payload && preflight.isSuccess && !preflight.isError);

  return (
    <>
      <Topbar
        eyebrow="Workspace"
        title="Start"
        subtitle="A guided first run using safe catalog defaults and preflight before launch."
        actions={
          <Button asChild variant="ghost" size="sm">
            <Link to="/train">
              Advanced training <ArrowRight />
            </Link>
          </Button>
        }
      />

      <div className="px-5 py-5 space-y-4">
        <div className="rounded-lg border border-border bg-surface px-4 py-3">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <div className="flex items-center gap-2">
                <Rocket className="h-4 w-4 text-accent" />
                <span className="text-sm font-semibold text-fg">First successful run</span>
              </div>
              <p className="mt-1 text-[13px] text-fg-muted">
                Halo Forge will use a small catalog model, a known dataset, and conservative
                SFT settings so the first launch proves the machine and data path.
              </p>
            </div>
            <Button
              variant="primary"
              size="md"
              disabled={!ready || launch.isPending}
              onClick={() => payload && launch.mutate(payload)}
            >
              {launch.isPending ? <Loader2 className="animate-spin" /> : <Play />}
              Launch first run
            </Button>
          </div>
        </div>

        <div className="grid gap-3 lg:grid-cols-3">
          <StepCard
            step="01"
            title="Backend"
            ready={Boolean(backend.data)}
            loading={backend.isLoading}
            body={backend.data ? `${backend.data.name} · ${backend.data.device}` : "Detecting accelerator"}
          />
          <StepCard
            step="02"
            title="Model"
            ready={Boolean(recommended)}
            loading={models.isLoading}
            body={recommended ? recommended.id : "Selecting a first-run catalog model"}
            detail={
              recommended
                ? `${recommended.memory_tier} · ~${recommended.estimated_memory_gb ?? "?"}GB · ${recommended.risk_level}`
                : undefined
            }
          />
          <StepCard
            step="03"
            title="Preflight"
            ready={ready}
            loading={preflight.isPending}
            body={
              preflight.isError
                ? (preflight.error as Error).message
                : preflight.isSuccess
                  ? "Launch checks passed"
                  : "Waiting for generated launch payload"
            }
            tone={preflight.isError ? "danger" : ready ? "success" : "neutral"}
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
            {runId ? (
              <div className="flex items-center justify-between rounded-md border border-success/30 bg-success-bg px-3 py-2">
                <span className="font-mono text-[11px] text-success">Started {runId}</span>
                <Button asChild size="sm" variant="ghost">
                  <Link to="/runs/$runId" params={{ runId }}>
                    Open run <ArrowRight />
                  </Link>
                </Button>
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
    ["Model", String(payload.model)],
    ["Dataset", String(payload.dataset)],
    ["Memory", model?.estimated_memory_gb ? `~${model.estimated_memory_gb}GB` : "unknown"],
    ["Output", String(payload.output_dir)],
  ];
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

function selectFirstRunModel(items: ModelCatalogEntry[]): ModelCatalogEntry | null {
  return items.find((item) => item.recommended_first_run && item.risk_level === "safe") ?? items[0] ?? null;
}

function extractRunId(data: unknown): string | null {
  if (!data || typeof data !== "object") return null;
  const record = data as Record<string, unknown>;
  return String(record.run_id ?? record.id ?? "").trim() || null;
}

function slug(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 32);
}
