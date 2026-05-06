import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import {
  ArrowLeft,
  Cpu,
  Layers,
  RefreshCw,
  Target,
  TrendingDown,
} from "lucide-react";
import { useMemo } from "react";
import { api, type CycleMetric, type RunDetail } from "@/lib/api";
import { queryKeys } from "@/lib/hooks";
import { Topbar } from "@/components/shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardEyebrow,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { MetricChart, type MetricSeries } from "@/components/charts/metric-chart";
import { cn, relativeTime } from "@/lib/utils";

export const Route = createFileRoute("/runs/$runId")({
  component: RunDetailRoute,
});

/**
 * Run detail view (phase D v1).
 *
 * Anatomy (top to bottom):
 *
 *   Topbar       run identity + status pill + back/refresh
 *   Stat ribbon  cycles · final loss · weights · effectiveness verdict
 *   Two-col      Loss & reward charts (2/3) | Run summary card (1/3)
 *   Cycle table  Per-cycle metrics with sparklines per row
 *
 * v1 ships static cycle metrics from the run summary. v2 will wire SSE
 * for active runs (live tail of train_loss while a job is running),
 * sample inspector, and verifier-breakdown histograms.
 */

function RunDetailRoute() {
  const { runId } = Route.useParams();
  const detailQuery = useQuery<RunDetail>({
    queryKey: queryKeys.runDetail(runId),
    queryFn: () => api.runDetail(runId),
    refetchInterval: 5_000,
    refetchIntervalInBackground: false,
  });
  const data = detailQuery.data;

  const cycleMetrics = useMemo<CycleMetric[]>(() => {
    return (data?.details?.cycle_metrics ?? []) as CycleMetric[];
  }, [data]);

  const isLive = isJobRunning(data?.status);

  return (
    <>
      <Topbar
        eyebrow={data ? `Runs / ${String(data.modality)}` : "Runs"}
        title={runId}
        subtitle={data?.model_name ?? undefined}
        live={isLive}
        actions={
          <>
            <Button variant="ghost" size="icon" asChild aria-label="Back to runs">
              <Link to="/runs">
                <ArrowLeft />
              </Link>
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => detailQuery.refetch()}
              aria-label="Refresh"
              disabled={detailQuery.isFetching}
            >
              <RefreshCw className={detailQuery.isFetching ? "animate-spin" : undefined} />
            </Button>
          </>
        }
        statusBar={
          <>
            <ReadoutItem label="STATUS" value={String(data?.status ?? "—")} />
            <ReadoutSep />
            <ReadoutItem
              label="STARTED"
              value={data?.timestamp ? relativeTime(String(data.timestamp)) : "—"}
            />
            <ReadoutSep />
            <ReadoutItem
              label="CYCLES"
              value={String(data?.details?.cycles_executed ?? 0)}
            />
            <ReadoutSep />
            <ReadoutItem
              label="STEPS"
              value={String(data?.metrics_summary?.update_steps ?? "—")}
            />
          </>
        }
      />

      {detailQuery.isLoading ? (
        <LoadingState />
      ) : detailQuery.isError ? (
        <ErrorState message={(detailQuery.error as Error).message} />
      ) : !data ? (
        <ErrorState message="Run not found." />
      ) : (
        <div className="px-5 py-5 space-y-4">
          <StatRibbon data={data} />

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
            <div className="lg:col-span-2 space-y-3">
              <LossCard cycles={cycleMetrics} />
              <RewardCard cycles={cycleMetrics} modality={String(data.modality)} />
            </div>
            <div className="space-y-3">
              <RunSummaryCard data={data} />
              <YieldCard yieldData={data.details?.yield_diagnostics} />
            </div>
          </div>

          <CycleTable cycles={cycleMetrics} modality={String(data.modality)} />
        </div>
      )}
    </>
  );
}

/* -------------------------------------------------------------------------
 * Stat ribbon
 * ----------------------------------------------------------------------- */

function StatRibbon({ data }: { data: RunDetail }) {
  // The detail endpoint stashes headline numbers in `metrics_summary`
  // and the verdict tone in `user_summary.confidence_tone`. Tiles read
  // from those structured fields rather than guessing at top-level keys.
  const finalLoss = data.metrics_summary?.final_train_loss;
  const updateSteps = data.metrics_summary?.update_steps ?? 0;
  const tone = (data.user_summary?.confidence_tone ?? "neutral") as
    | "success"
    | "warning"
    | "danger"
    | "neutral";
  const verdict =
    tone === "success"
      ? "passed"
      : tone === "danger"
        ? "failed"
        : tone === "warning"
          ? "review"
          : data.status || "—";

  const tiles: Array<{ label: string; value: string; hint?: string; tone?: typeof tone }> = [
    {
      label: "FINAL LOSS",
      value: typeof finalLoss === "number" ? finalLoss.toFixed(3) : "—",
    },
    {
      label: "CYCLES",
      value: String(data.details?.cycles_executed ?? "—"),
      hint: data.details?.cycles_executed
        ? `${data.details.cycles_executed} executed`
        : undefined,
    },
    {
      label: "STEPS",
      value: String(updateSteps),
      tone: updateSteps > 0 ? "success" : "warning",
    },
    {
      label: "VERDICT",
      value: verdict,
      tone,
    },
  ];

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-2.5">
      {tiles.map((t) => (
        <Card key={t.label}>
          <CardContent className="px-3.5 py-3">
            <div className="text-[10px] font-medium uppercase tracking-[0.14em] text-fg-disabled">
              {t.label}
            </div>
            <div className="mt-1.5 flex items-baseline gap-2">
              <span
                className={cn(
                  "font-mono text-[22px] leading-none tabular-nums tracking-tight",
                  t.tone === "success"
                    ? "text-success"
                    : t.tone === "warning"
                      ? "text-warning"
                      : t.tone === "danger"
                        ? "text-danger"
                        : "text-fg",
                )}
              >
                {t.value}
              </span>
              {t.hint ? (
                <span className="font-mono text-[11px] text-fg-subtle">{t.hint}</span>
              ) : null}
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

/* -------------------------------------------------------------------------
 * Charts
 * ----------------------------------------------------------------------- */

function LossCard({ cycles }: { cycles: CycleMetric[] }) {
  const series: MetricSeries[] = [
    {
      key: "train_loss",
      label: "Train loss",
      tone: "accent",
      format: (v) => v.toFixed(4),
    },
    {
      key: "eval_loss",
      label: "Eval loss",
      tone: "info",
      format: (v) => v.toFixed(4),
    },
  ];
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardEyebrow>METRICS</CardEyebrow>
          <CardTitle>Loss</CardTitle>
          <TrendingDown className="h-3.5 w-3.5 text-fg-disabled" />
        </div>
        <Legend series={series} />
      </CardHeader>
      <CardContent className="px-3 py-4">
        <MetricChart
          data={cycles}
          xKey="cycle"
          series={series}
          height={200}
          yFormat={(v) => v.toFixed(2)}
          emptyState="Loss metrics will appear here once the first cycle completes."
        />
      </CardContent>
    </Card>
  );
}

function RewardCard({
  cycles,
  modality,
}: {
  cycles: CycleMetric[];
  modality: string;
}) {
  // Reward is RAFT-specific; SFT runs don't populate it.
  if (modality !== "raft") return null;

  const series: MetricSeries[] = [
    {
      key: "avg_reward",
      label: "Avg reward",
      tone: "muted",
      format: (v) => v.toFixed(3),
    },
    {
      key: "avg_kept_reward",
      label: "Avg kept reward",
      tone: "accent",
      format: (v) => v.toFixed(3),
    },
    {
      key: "success_rate",
      label: "Success rate",
      tone: "success",
      format: (v) => `${(v * 100).toFixed(1)}%`,
    },
  ];

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardEyebrow>RLVR</CardEyebrow>
          <CardTitle>Reward distribution</CardTitle>
          <Target className="h-3.5 w-3.5 text-fg-disabled" />
        </div>
        <Legend series={series} />
      </CardHeader>
      <CardContent className="px-3 py-4">
        <MetricChart
          data={cycles}
          xKey="cycle"
          series={series}
          height={180}
          yFormat={(v) => v.toFixed(2)}
          emptyState="Reward metrics appear once the verifier scores its first cycle."
        />
      </CardContent>
    </Card>
  );
}

function Legend({ series }: { series: MetricSeries[] }) {
  return (
    <div className="flex items-center gap-3">
      {series.map((s) => (
        <span
          key={s.key}
          className="inline-flex items-center gap-1.5 text-[11px] text-fg-muted"
        >
          <span
            className="status-dot"
            style={{ background: `var(--color-${s.tone ?? "accent"})` }}
          />
          {s.label}
        </span>
      ))}
    </div>
  );
}

/* -------------------------------------------------------------------------
 * Side cards
 * ----------------------------------------------------------------------- */

function RunSummaryCard({ data }: { data: RunDetail }) {
  const rows: Array<{ label: string; value: string; mono?: boolean }> = [
    { label: "Modality", value: String(data.modality ?? "—") },
    { label: "Model", value: String(data.model_name ?? "—"), mono: true },
    { label: "Status", value: String(data.status ?? "—") },
    {
      label: "Started",
      value: data.timestamp ? relativeTime(String(data.timestamp)) : "—",
    },
    {
      label: "Cycles executed",
      value: String(data.details?.cycles_executed ?? "—"),
      mono: true,
    },
    {
      label: "Seed",
      value: String(data.details?.seed ?? "—"),
      mono: true,
    },
    {
      label: "Final model",
      value: data.details?.final_model_available ? "saved" : "—",
    },
  ];

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardEyebrow>RUN</CardEyebrow>
          <CardTitle>Summary</CardTitle>
        </div>
        <Layers className="h-3.5 w-3.5 text-fg-disabled" />
      </CardHeader>
      <CardContent className="p-0 divide-y divide-border-subtle text-[12.5px]">
        {rows.map((r) => (
          <div key={r.label} className="flex items-start justify-between gap-3 px-3.5 py-2">
            <span className="text-fg-muted">{r.label}</span>
            <span
              className={cn(
                "text-right truncate max-w-[20ch]",
                r.mono ? "font-mono text-fg" : "text-fg",
              )}
              title={r.value}
            >
              {r.value}
            </span>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

function YieldCard({
  yieldData,
}: {
  yieldData?: Record<string, unknown>;
}) {
  if (!yieldData) return null;

  const rates = (yieldData.rates ?? {}) as Record<string, number | undefined>;
  const stages = (yieldData.stage_counts ?? {}) as Record<string, number | undefined>;
  const summary = (yieldData.summary ?? {}) as { status?: string; text?: string };

  const tone =
    summary.status === "healthy"
      ? "success"
      : summary.status === "low_yield"
        ? "warning"
        : summary.status === "no_signal"
          ? "danger"
          : "neutral";

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardEyebrow>QUALITY</CardEyebrow>
          <CardTitle>Yield</CardTitle>
        </div>
        {summary.status ? (
          <Badge tone={tone} dot size="sm">
            {summary.status}
          </Badge>
        ) : null}
      </CardHeader>
      <CardContent className="space-y-2.5">
        {summary.text ? (
          <p className="text-[12px] text-fg-muted leading-relaxed">{summary.text}</p>
        ) : null}
        <div className="grid grid-cols-3 gap-2 text-center">
          {(["generated", "verified", "kept"] as const).map((stage) => (
            <div key={stage} className="rounded-md border border-border-subtle px-2 py-2">
              <div className="text-[10px] uppercase tracking-wider text-fg-disabled font-medium">
                {stage}
              </div>
              <div className="font-mono text-sm tabular-nums text-fg mt-0.5">
                {stages[stage] ?? "—"}
              </div>
            </div>
          ))}
        </div>
        <div className="space-y-1 pt-1">
          {Object.entries(rates).map(([k, v]) => (
            <div key={k} className="flex items-center justify-between text-[11.5px]">
              <span className="text-fg-muted">{k.replace(/_/g, " ")}</span>
              <span className="font-mono text-fg">
                {typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "—"}
              </span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

/* -------------------------------------------------------------------------
 * Cycle table
 * ----------------------------------------------------------------------- */

function CycleTable({
  cycles,
  modality,
}: {
  cycles: CycleMetric[];
  modality: string;
}) {
  if (!cycles.length) return null;
  const isRaft = modality === "raft";

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardEyebrow>TIMELINE</CardEyebrow>
          <CardTitle>Per-cycle metrics</CardTitle>
        </div>
        <span className="text-[11px] text-fg-subtle">{cycles.length} cycles</span>
      </CardHeader>
      <CardContent className="p-0">
        <table className="w-full text-[12.5px]">
          <thead>
            <tr className="border-b border-border-subtle">
              <Th>Cycle</Th>
              <Th align="right">Train loss</Th>
              <Th align="right">Eval loss</Th>
              {isRaft ? <Th align="right">Avg reward</Th> : null}
              {isRaft ? <Th align="right">Kept</Th> : null}
              {isRaft ? <Th align="right">Success</Th> : null}
              <Th align="right">Steps</Th>
              <Th align="right">Duration</Th>
            </tr>
          </thead>
          <tbody>
            {cycles.map((c) => (
              <tr
                key={c.cycle}
                className="border-b border-border-subtle last:border-0 hover:bg-surface-hover/30 transition-colors"
              >
                <Td mono>#{c.cycle}</Td>
                <Td align="right" mono>
                  {fmt(c.train_loss, 4)}
                </Td>
                <Td align="right" mono>
                  {fmt(c.eval_loss, 4)}
                </Td>
                {isRaft ? (
                  <Td align="right" mono>
                    {fmt(c.avg_reward, 3)}
                  </Td>
                ) : null}
                {isRaft ? (
                  <Td align="right" mono className="text-fg-muted">
                    {c.samples_kept != null && c.samples_seen != null
                      ? `${c.samples_kept}/${c.samples_seen}`
                      : "—"}
                  </Td>
                ) : null}
                {isRaft ? (
                  <Td align="right" mono>
                    {c.success_rate != null
                      ? `${(c.success_rate * 100).toFixed(1)}%`
                      : "—"}
                  </Td>
                ) : null}
                <Td align="right" mono className="text-fg-muted">
                  {c.train_steps_executed ?? "—"}
                </Td>
                <Td align="right" mono className="text-fg-muted">
                  {fmtDuration(c.cycle_duration_seconds)}
                </Td>
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}

function Th({
  children,
  align = "left",
}: {
  children: React.ReactNode;
  align?: "left" | "right";
}) {
  return (
    <th
      className={cn(
        "px-3.5 py-2 text-[10px] font-medium uppercase tracking-[0.12em] text-fg-disabled",
        align === "right" ? "text-right" : "text-left",
      )}
    >
      {children}
    </th>
  );
}

function Td({
  children,
  align = "left",
  mono,
  className,
}: {
  children: React.ReactNode;
  align?: "left" | "right";
  mono?: boolean;
  className?: string;
}) {
  return (
    <td
      className={cn(
        "px-3.5 py-2",
        align === "right" ? "text-right" : "text-left",
        mono && "font-mono tabular-nums",
        className,
      )}
    >
      {children}
    </td>
  );
}

/* -------------------------------------------------------------------------
 * States + helpers
 * ----------------------------------------------------------------------- */

function LoadingState() {
  return (
    <div className="px-5 py-5 space-y-3">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-2.5">
        {[0, 1, 2, 3].map((i) => (
          <div
            key={i}
            className="h-20 animate-pulse rounded-lg border border-border bg-surface/40"
          />
        ))}
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        <div className="lg:col-span-2 space-y-3">
          <div className="h-64 animate-pulse rounded-lg border border-border bg-surface/40" />
          <div className="h-48 animate-pulse rounded-lg border border-border bg-surface/40" />
        </div>
        <div className="h-72 animate-pulse rounded-lg border border-border bg-surface/40" />
      </div>
    </div>
  );
}

function ErrorState({ message }: { message: string }) {
  return (
    <div className="px-5 py-12">
      <Card>
        <CardContent className="text-center space-y-2 py-12">
          <Cpu className="h-6 w-6 text-fg-subtle mx-auto" />
          <div className="text-sm font-medium text-fg">Run unavailable</div>
          <div className="text-xs text-fg-muted max-w-[40ch] mx-auto">{message}</div>
          <Button asChild variant="ghost" size="sm" className="mt-2">
            <Link to="/runs">Back to runs</Link>
          </Button>
        </CardContent>
      </Card>
    </div>
  );
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

function fmt(v: number | null | undefined, digits: number): string {
  if (v == null || Number.isNaN(v)) return "—";
  return v.toFixed(digits);
}

function fmtDuration(seconds: number | null | undefined): string {
  if (seconds == null || seconds <= 0) return "—";
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}

function isJobRunning(status: string | undefined): boolean {
  if (!status) return false;
  const s = status.toLowerCase();
  return s === "running" || s === "active" || s === "in_progress" || s === "pending";
}
