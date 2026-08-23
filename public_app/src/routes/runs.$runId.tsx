import { createFileRoute, Link } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowLeft,
  Cpu,
  CheckCircle2,
  CircleAlert,
  Copy,
  Database,
  FlaskConical,
  Layers,
  Loader2,
  Pin,
  PinOff,
  Plug,
  RefreshCw,
  Square,
  Target,
  TrendingDown,
  X,
  Zap,
} from "lucide-react";
import { useMemo, useState } from "react";
import {
  api,
  type CycleMetric,
  type DatasetBinding,
  type Evaluation,
  type ModelArtifactOccurrence,
  type RunCost,
  type RunDetail,
  type RunLive,
} from "@/lib/api";
import { useEventSource } from "@/lib/event-source";
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
import { CycleScrubber } from "@/components/run/cycle-scrubber";
import { LogsPanel } from "@/components/run/logs-panel";
import { SampleInspector } from "@/components/run/sample-inspector";
import { RunCheckpointTrajectory } from "@/components/research/adaptive-workspace";
import { RunIntegrityStrip, RunRewardAuditWorkspace } from "@/components/research/reward-integrity-workspace";
import {
  PINNED_RUNS_LIMIT,
  pinRun,
  unpinRun,
  usePinnedRuns,
} from "@/lib/pinned-runs";
import { cn, relativeTime } from "@/lib/utils";

type RunTaskTab = "monitor" | "metrics" | "data" | "evaluation" | "artifacts" | "logs";

const RUN_TASK_TABS: Array<{ id: RunTaskTab; label: string }> = [
  { id: "monitor", label: "Monitor" },
  { id: "metrics", label: "Metrics" },
  { id: "data", label: "Data" },
  { id: "evaluation", label: "Evaluation" },
  { id: "artifacts", label: "Artifacts" },
  { id: "logs", label: "Logs" },
];

export const Route = createFileRoute("/runs/$runId")({
  validateSearch: (search: Record<string, unknown>): { tab?: RunTaskTab; evidence?: "evaluations" | "training-audits"; audit?: string; sample?: string; page?: number; classification?: string } => ({
    tab: isRunTaskTab(search.tab) ? search.tab : undefined,
    evidence: search.evidence === "training-audits" ? "training-audits" : search.evidence === "evaluations" ? "evaluations" : undefined,
    audit: typeof search.audit === "string" ? search.audit : undefined,
    sample: typeof search.sample === "string" ? search.sample : undefined,
    page: typeof search.page === "number" && Number.isFinite(search.page) ? Math.max(1, Math.floor(search.page)) : undefined,
    classification: typeof search.classification === "string" ? search.classification : undefined,
  }),
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
 * Active runs use the SSE/live payload for stage, progress, and metric
 * history; completed runs fall back to chart-ready summary artifacts.
 */

function RunDetailRoute() {
  const { runId } = Route.useParams();
  const search = Route.useSearch();
  const navigate = Route.useNavigate();
  const activeTab = search.tab ?? "monitor";
  const queryClient = useQueryClient();
  const detailQuery = useQuery<RunDetail>({
    queryKey: queryKeys.runDetail(runId),
    queryFn: () => api.runDetail(runId),
    refetchInterval: 5_000,
    refetchIntervalInBackground: false,
  });
  const data = detailQuery.data;
  const evaluationsQuery = useQuery({
    queryKey: ["evaluations", "run", runId],
    queryFn: () => api.listEvaluations({ runId }),
    enabled: Boolean(data),
    refetchInterval: 10_000,
    refetchIntervalInBackground: false,
    retry: false,
  });
  const artifactsQuery = useQuery({
    queryKey: ["artifacts", "run", runId],
    queryFn: () => api.listModelArtifacts({ runId, limit: 100 }),
    enabled: Boolean(data) && activeTab === "artifacts",
    retry: false,
  });
  const detailStatus = data?.status ?? "";
  const detailIsLive = isJobRunning(detailStatus);
  const liveStream = useEventSource<RunLive>(
    detailIsLive ? `/api/public/runs/${encodeURIComponent(runId)}/events` : null,
  );
  const live = liveStream.data;
  const displayStatus = live?.status ?? data?.status;

  const cycleMetrics = useMemo<CycleMetric[]>(() => {
    return (data?.details?.cycle_metrics ?? []) as CycleMetric[];
  }, [data]);
  const liveMetricPoints = useMemo(() => live?.metric_points ?? [], [live]);

  const isLive = isJobRunning(displayStatus);

  // Phase F — pinning. Pinned runs accumulate in localStorage and feed
  // the comparison route at /runs/compare. Cap is enforced by the store
  // itself; we surface a disabled state once the user is at the cap so
  // the action button doesn't lie about what it'll do.
  const pinnedIds = usePinnedRuns();
  const isPinned = pinnedIds.includes(runId);
  const pinDisabled = !isPinned && pinnedIds.length >= PINNED_RUNS_LIMIT;

  // Phase D v3 — chart focus state. `null` means "show all cycles";
  // a number means "slice the chart to cycles 0..focusCycle inclusive".
  // The CycleScrubber reads/writes this; chart cards consume the
  // sliced view via slicedCycles below.
  const [focusCycle, setFocusCycle] = useState<number | null>(null);
  const slicedCycles = useMemo<CycleMetric[]>(() => {
    if (focusCycle === null) return cycleMetrics;
    return cycleMetrics.filter((c) => c.cycle <= focusCycle);
  }, [cycleMetrics, focusCycle]);

  // Cancel button — only meaningful for active jobs. Mutation invalidates
  // the run-detail query so the topbar status pulls the post-cancel state.
  const cancelMutation = useMutation({
    mutationFn: () => api.runCancel(runId),
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.runDetail(runId) });
    },
  });

  return (
    <>
      <Topbar
        eyebrow={data ? `Runs / ${String(data.modality)}` : "Runs"}
        title={runId}
        subtitle={data?.model_name ?? undefined}
        live={isLive}
        actions={
          <>
            {isLive ? (
              <Button
                variant="danger"
                size="sm"
                onClick={() => cancelMutation.mutate()}
                disabled={cancelMutation.isPending}
                title="Send SIGTERM and wait for graceful checkpoint save"
              >
                {cancelMutation.isPending ? (
                  <Loader2 className="animate-spin" />
                ) : (
                  <Square />
                )}
                Cancel run
              </Button>
            ) : null}
            {!isLive && data ? (
              <Button variant="ghost" size="sm" asChild>
                <Link
                  to="/train"
                  search={{
                    template: undefined,
                    model: undefined,
                    mode: undefined,
                    datasetVersion: undefined,
                    datasetSplit: undefined,
                    parentRun: runId,
                  }}
                >
                  <Copy />Clone in Train
                </Link>
              </Button>
            ) : null}
            {!isLive && data ? (
              <Button variant="primary" size="sm" asChild>
                <Link to="/eval" search={{ runId, suite: undefined, evaluation: undefined }}>
                  <FlaskConical />Evaluate
                </Link>
              </Button>
            ) : null}
            {data ? (
              <Button variant="ghost" size="sm" asChild>
                <Link to="/datasets/review" search={{ new: "1", source: "run_samples", sourceRef: runId, baseRef: undefined }}>
                  <CheckCircle2 />Review samples
                </Link>
              </Button>
            ) : null}
            <Button
              variant={isPinned ? "primary" : "ghost"}
              size="sm"
              onClick={() => (isPinned ? unpinRun(runId) : pinRun(runId))}
              disabled={pinDisabled}
              title={
                isPinned
                  ? "Unpin from comparison tray"
                  : pinDisabled
                    ? `Comparison tray full (${PINNED_RUNS_LIMIT} runs max)`
                    : "Pin to comparison tray"
              }
              aria-pressed={isPinned}
            >
              {isPinned ? <PinOff /> : <Pin />}
              {isPinned ? "Pinned" : "Pin"}
            </Button>
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
            <ReadoutItem label="STATUS" value={plainRunStatus(displayStatus, liveStream.status, liveStream.error)} />
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

      <RunTaskTabs runId={runId} active={activeTab} />

      {detailQuery.isLoading ? (
        <LoadingState />
      ) : detailQuery.isError ? (
        <ErrorState message={(detailQuery.error as Error).message} />
      ) : !data ? (
        <ErrorState message="Run not found." />
      ) : (
        <div className="px-5 py-5 space-y-4">
          {activeTab === "monitor" ? (
            <>
              <LiveSummary data={data} live={live} streamStatus={liveStream.status} streamError={liveStream.error} />
              <RunIntegrityStrip runId={runId} />
              {isLive || live?.stage ? <StageRail live={live} status={displayStatus} /> : null}
              {(data.run_group_id || data.details?.run_group_id) ? <RunCheckpointTrajectory runGroupId={String(data.run_group_id || data.details?.run_group_id)} runId={runId} /> : null}
              {data.failure_summary ? <FailureSummaryCard data={data.failure_summary} /> : null}
              <StatRibbon data={data} />
              <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
                <div className="lg:col-span-2"><LiveNowCard data={data} live={live} /></div>
                <div className="space-y-3">
                  <RunSummaryCard data={data} />
                  <CostCard cost={data.details?.cost} />
                  <YieldCard yieldData={data.details?.yield_diagnostics} />
                </div>
              </div>
            </>
          ) : null}

          {activeTab === "metrics" ? (
            <>
              {cycleMetrics.length > 1 ? (
                <Card><CardContent className="px-4 py-2.5"><CycleScrubber cycles={cycleMetrics.map((c) => c.cycle)} focus={focusCycle} onFocusChange={setFocusCycle} /></CardContent></Card>
              ) : null}
              <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
                <LossCard cycles={slicedCycles} livePoints={liveMetricPoints} stage={live?.stage ?? null} />
                <RewardCard cycles={slicedCycles} modality={String(data.modality)} />
              </div>
              <CycleTable cycles={cycleMetrics} modality={String(data.modality)} />
            </>
          ) : null}

          {activeTab === "data" ? (
            <div className="grid grid-cols-1 gap-3 xl:grid-cols-[minmax(280px,0.7fr)_minmax(460px,1.3fr)]">
              <DatasetBindingsCard bindings={data.datasets ?? []} />
              <SampleInspector runId={runId} availableCycles={cycleMetrics.map((c) => c.cycle)} enabled={String(data.modality) === "raft"} />
            </div>
          ) : null}

          {activeTab === "evaluation" ? (
            <div className="overflow-hidden rounded-lg border border-border bg-bg">
              <nav className="flex gap-1 overflow-x-auto border-b border-border bg-bg-subtle/35 px-3" aria-label="Run evidence">
                <Link to="/runs/$runId" params={{ runId }} search={{ tab: "evaluation", evidence: "evaluations" }} className={cn("relative h-10 px-3 text-[10.5px] leading-10", (search.evidence ?? "evaluations") === "evaluations" ? "font-medium text-fg" : "text-fg-subtle hover:text-fg")}>Evaluations{(search.evidence ?? "evaluations") === "evaluations" ? <span className="absolute inset-x-2 bottom-0 h-0.5 bg-accent" /> : null}</Link>
                <Link to="/runs/$runId" params={{ runId }} search={{ tab: "evaluation", evidence: "training-audits" }} className={cn("relative h-10 px-3 text-[10.5px] leading-10", search.evidence === "training-audits" ? "font-medium text-fg" : "text-fg-subtle hover:text-fg")}>Training audits{search.evidence === "training-audits" ? <span className="absolute inset-x-2 bottom-0 h-0.5 bg-accent" /> : null}</Link>
              </nav>
              {search.evidence === "training-audits" ? <RunRewardAuditWorkspace runId={runId} selectedAuditId={search.audit} selectedSampleId={search.sample} page={search.page ?? 1} classification={search.classification} onAudit={(audit) => navigate({ to: "/runs/$runId", params: { runId }, search: { ...search, tab: "evaluation", evidence: "training-audits", audit, sample: undefined, page: 1 }, replace: true })} onSample={(sample) => navigate({ to: "/runs/$runId", params: { runId }, search: { ...search, sample }, replace: true })} onPage={(page) => navigate({ to: "/runs/$runId", params: { runId }, search: { ...search, page, sample: undefined }, replace: true })} onClassification={(classification) => navigate({ to: "/runs/$runId", params: { runId }, search: { ...search, classification, page: 1, sample: undefined }, replace: true })} /> : <EvaluationHistoryCard runId={runId} items={evaluationsQuery.data?.items ?? data.evaluations ?? []} loading={evaluationsQuery.isLoading} />}
            </div>
          ) : null}

          {activeTab === "artifacts" ? (
            <div className="grid grid-cols-1 gap-3 xl:grid-cols-[minmax(460px,1.4fr)_minmax(280px,0.6fr)]">
              <RunArtifactsPanel items={artifactsQuery.data?.items ?? []} loading={artifactsQuery.isLoading} error={artifactsQuery.isError ? (artifactsQuery.error as Error).message : null} finalModelAvailable={Boolean(data.details?.final_model_available)} />
              <LineageCard runId={runId} />
            </div>
          ) : null}

          {activeTab === "logs" ? <LogsPanel runId={runId} tail={500} height={560} /> : null}
        </div>
      )}
    </>
  );
}

function RunTaskTabs({ runId, active }: { runId: string; active: RunTaskTab }) {
  return (
    <nav aria-label="Run detail" className="sticky top-[49px] z-10 flex overflow-x-auto border-b border-border bg-bg-subtle/95 px-3 backdrop-blur md:px-5">
      {RUN_TASK_TABS.map((tab) => (
        <Link
          key={tab.id}
          to="/runs/$runId"
          params={{ runId }}
          search={{ tab: tab.id }}
          aria-current={active === tab.id ? "page" : undefined}
          className={cn("relative flex h-11 shrink-0 items-center px-3 text-[12px] transition-colors", active === tab.id ? "font-medium text-fg" : "text-fg-subtle hover:text-fg")}
        >
          {tab.label}
          {active === tab.id ? <span className="absolute inset-x-2 bottom-0 h-0.5 rounded-full bg-accent" /> : null}
        </Link>
      ))}
    </nav>
  );
}

function RunArtifactsPanel({ items, loading, error, finalModelAvailable }: { items: ModelArtifactOccurrence[]; loading: boolean; error: string | null; finalModelAvailable: boolean }) {
  return (
    <Card>
      <CardHeader><div><CardEyebrow>OUTPUTS</CardEyebrow><CardTitle>Published artifacts</CardTitle></div><Badge tone={items.length ? "success" : "neutral"} size="sm">{items.length}</Badge></CardHeader>
      <CardContent className="p-0">
        {loading ? <div className="flex items-center gap-2 px-4 py-8 text-xs text-fg-muted"><Loader2 className="h-4 w-4 animate-spin" />Loading artifact occurrences</div> : error ? <div className="px-4 py-6 text-xs text-danger">{error}</div> : items.length ? (
          <div className="divide-y divide-border-subtle">
            {items.map((item) => (
              <Link key={item.id} to="/models" search={{ tab: "artifacts", artifact: item.id }} className="flex items-center justify-between gap-3 px-4 py-3 hover:bg-bg-subtle/60">
                <div className="min-w-0"><div className="truncate text-xs font-medium text-fg">{runArtifactTitle(item)}</div><div className="mt-1 truncate font-mono text-[10px] text-fg-disabled">{item.content_hash || item.id}</div></div>
                <div className="shrink-0 text-right"><Badge tone={item.integrity === "verified" ? "success" : "neutral"} size="sm">{item.integrity || "indexed"}</Badge><div className="mt-1 text-[10px] text-fg-disabled">{[item.format, item.quantization || item.dtype].filter(Boolean).join(" · ")}</div></div>
              </Link>
            ))}
          </div>
        ) : <div className="px-4 py-8 text-center text-xs text-fg-muted">{finalModelAvailable ? "The final model is available but has not been indexed into the artifact library yet." : "No checkpoint or final artifact has been published for this run."}</div>}
      </CardContent>
    </Card>
  );
}

function runArtifactTitle(item: ModelArtifactOccurrence): string {
  const modelName = item.model_name?.trim();
  const isFilesystemPath = Boolean(modelName && (/^(?:[A-Za-z]:[\\/]|[\\/])/.test(modelName)));
  if (modelName && !isFilesystemPath) return modelName;
  return item.kind
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function isRunTaskTab(value: unknown): value is RunTaskTab {
  return RUN_TASK_TABS.some((tab) => tab.id === value);
}

function LiveSummary({
  data,
  live,
  streamStatus,
  streamError,
}: {
  data: RunDetail;
  live: RunLive | null;
  streamStatus: "idle" | "connecting" | "open" | "closed" | "error";
  streamError: string | null;
}) {
  const status = live?.status ?? data.status;
  const normalizedStatus = String(status ?? "").toLowerCase();
  const isTerminal = ["completed", "failed", "stopped", "cancelled", "canceled"].includes(
    normalizedStatus,
  );
  const failedTerminal = normalizedStatus === "failed";
  const loss = live?.latest_loss ?? data.metrics_summary?.final_train_loss ?? null;
  const steps = live?.current_step ?? data.metrics_summary?.update_steps ?? null;
  const totalSteps = live?.total_steps ?? null;
  const cycles = live?.current_cycle ?? data.details?.cycles_executed ?? null;
  const totalCycles = live?.total_cycles ?? null;
  const stage = live?.stage;
  const artifactState = live?.artifact_state ?? (data.details?.final_model_available ? "final_model" : "none");
  const nextStep =
    live?.next_step ??
    live?.user_summary?.next_step ??
    data.user_summary?.next_step ??
    data.next_step ??
    (isTerminal ? "Review outputs and compare the run." : "Keep this page open to monitor progress.");
  const headline =
    live?.headline ??
    live?.user_summary?.headline ??
    data.user_summary?.headline ??
    data.headline ??
    (isTerminal ? "Run finished" : "Run monitor");
  const streamLabel = plainRunStatus(status, streamStatus, streamError);
  const streamTone =
    failedTerminal
      ? "danger"
      : streamStatus === "open" || isTerminal
      ? "success"
      : streamStatus === "error"
        ? "warning"
        : "neutral";
  const methodSummary = summaryForMethod(String(data.modality ?? ""), data);
  const stageCopy = stage?.message ?? (isTerminal ? nextStep : "Waiting for the next training event.");

  return (
    <Card className="bg-surface/90">
      <CardContent className="flex flex-col gap-4 p-4 lg:flex-row lg:items-center lg:justify-between">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            {streamTone === "success" ? (
              <CheckCircle2 className="h-4 w-4 text-success" />
            ) : streamTone === "danger" ? (
              <CircleAlert className="h-4 w-4 text-danger" />
            ) : (
              <CircleAlert className="h-4 w-4 text-warning" />
            )}
            <span className="text-sm font-semibold text-fg">{headline}</span>
            <Badge tone={streamTone} dot size="sm">
              {streamLabel}
            </Badge>
          </div>
          <p className="mt-1 max-w-[72ch] text-[12.5px] leading-5 text-fg-muted">
            {methodSummary} {stageCopy}
          </p>
          {live?.last_event ? (
            <p className="mt-1 max-w-[72ch] truncate font-mono text-[11px] text-fg-disabled" title={live.last_event}>
              {live.last_event}
            </p>
          ) : null}
        </div>
        <div className="grid min-w-[360px] grid-cols-2 gap-2 lg:grid-cols-4">
          <LiveMetric label="Stage" value={stage?.label ?? String(data.modality ?? "-")} />
          <LiveMetric label="Elapsed" value={fmtDuration(live?.elapsed_seconds)} />
          <LiveMetric label="Loss" value={fmt(loss, 4)} />
          <LiveMetric label="Steps" value={formatProgress(steps, totalSteps)} />
          <LiveMetric label={cycleLikeMethod(String(data.modality ?? "")) ? "Cycles" : "Artifact"} value={cycleLikeMethod(String(data.modality ?? "")) ? formatProgress(cycles, totalCycles) : artifactStateLabel(artifactState)} />
        </div>
      </CardContent>
    </Card>
  );
}

const STAGE_ORDER = [
  "preparing",
  "loading_dataset",
  "downloading_model",
  "building_trainer",
  "training",
  "evaluating",
  "saving_checkpoint",
  "finalizing",
  "completed",
] as const;

const STAGE_LABELS: Record<string, string> = {
  preparing: "Prepare",
  loading_dataset: "Data",
  downloading_model: "Model",
  building_trainer: "Trainer",
  training: "Train",
  evaluating: "Eval",
  saving_checkpoint: "Save",
  finalizing: "Finalize",
  completed: "Done",
  failed: "Failed",
  cancelled: "Cancelled",
};

function StageRail({ live, status }: { live: RunLive | null; status?: string }) {
  const currentKey =
    live?.stage?.key ??
    (String(status ?? "").toLowerCase() === "running" ? "training" : String(status ?? "").toLowerCase());
  const terminalFailed = ["failed", "stopped", "cancelled", "canceled"].includes(currentKey);
  const activeIndex = STAGE_ORDER.indexOf(currentKey as (typeof STAGE_ORDER)[number]);
  const progress = live?.stage?.progress_percent ?? live?.progress_percent ?? 0;
  return (
    <Card>
      <CardContent className="space-y-3 px-4 py-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="text-[10px] font-medium uppercase tracking-[0.14em] text-fg-disabled">Live stage</div>
            <div className="mt-1 text-sm font-semibold text-fg">
              {live?.stage?.label ?? STAGE_LABELS[currentKey] ?? "Waiting"}
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-[11px] text-fg-muted">
            <Badge tone={terminalFailed ? "danger" : currentKey === "completed" ? "success" : "neutral"} size="sm">
              {Math.max(0, Math.min(100, Number(progress) || 0)).toFixed(0)}%
            </Badge>
            {live?.eta_seconds ? <span>ETA {fmtDuration(live.eta_seconds)}</span> : null}
            {live?.artifact_state ? <span>Artifact {artifactStateLabel(live.artifact_state)}</span> : null}
          </div>
        </div>
        <div className="grid grid-cols-3 gap-1.5 lg:grid-cols-9">
          {STAGE_ORDER.map((key, index) => {
            const done = currentKey === "completed" || (activeIndex >= 0 && index < activeIndex);
            const active = key === currentKey;
            return (
              <div
                key={key}
                className={cn(
                  "h-8 rounded-sm border px-2 py-1 text-[10px] leading-5",
                  done
                    ? "border-success/35 bg-success-bg/20 text-success"
                    : active
                      ? "border-accent/70 bg-accent/10 text-fg"
                      : "border-border-subtle bg-bg-subtle/40 text-fg-disabled",
                )}
              >
                {STAGE_LABELS[key]}
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}

function LiveNowCard({ data, live }: { data: RunDetail; live: RunLive | null }) {
  const loss = live?.latest_loss ?? data.metrics_summary?.final_train_loss ?? null;
  const steps = live?.current_step ?? data.metrics_summary?.update_steps ?? null;
  const totalSteps = live?.total_steps ?? null;
  const stageLabel = live?.stage?.label ?? plainRunStatus(data.status, "idle", null);
  const stageMessage =
    live?.stage?.message ??
    (String(data.status ?? "").toLowerCase() === "completed"
      ? "Training finished. Review the final artifact and results."
      : String(data.status ?? "").toLowerCase() === "failed"
        ? "Training stopped before completion. Review the failure summary and logs."
        : "Waiting for the next training event.");
  const artifact = live?.artifact_state ?? (data.details?.final_model_available ? "final_model" : "none");

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardEyebrow>LIVE</CardEyebrow>
          <CardTitle>Now</CardTitle>
          <Zap className="h-3.5 w-3.5 text-accent" />
        </div>
        <Badge tone={String(data.status).toLowerCase() === "failed" ? "danger" : "neutral"} size="sm">
          {stageLabel}
        </Badge>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-[12.5px] leading-5 text-fg-muted">{stageMessage}</p>
        <div className="grid grid-cols-2 gap-2">
          <LiveMetric label="Elapsed" value={fmtDuration(live?.elapsed_seconds)} />
          <LiveMetric label="Step" value={formatProgress(steps, totalSteps)} />
          <LiveMetric label="Loss" value={fmt(loss, 4)} />
          <LiveMetric label="Artifact" value={artifactStateLabel(artifact)} />
        </div>
        {live?.last_event ? (
          <div className="rounded-md border border-border-subtle bg-bg-subtle/50 px-3 py-2">
            <div className="text-[10px] uppercase tracking-[0.14em] text-fg-disabled">Latest event</div>
            <div className="mt-1 break-words font-mono text-[11px] leading-4 text-fg-muted">
              {live.last_event}
            </div>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}

function LiveMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border-subtle bg-bg-subtle/50 px-3 py-2">
      <div className="text-[10px] uppercase tracking-[0.14em] text-fg-disabled">
        {label}
      </div>
      <div className="mt-1 truncate font-mono text-[12px] text-fg" title={value}>
        {value}
      </div>
    </div>
  );
}

function FailureSummaryCard({ data }: { data: NonNullable<RunDetail["failure_summary"]> }) {
  const tail = data.log_tail?.filter(Boolean).slice(-8) ?? [];
  return (
    <Card className="border-danger/40 bg-danger-bg/20">
      <CardHeader>
        <div className="flex items-center gap-2">
          <CircleAlert className="h-4 w-4 text-danger" />
          <CardTitle>{data.headline}</CardTitle>
        </div>
        <Badge tone="danger" size="sm">{data.kind}</Badge>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="max-w-[82ch] text-[13px] leading-5 text-fg-muted">
          {data.message}
        </p>
        <div className="rounded-sm border border-danger/25 bg-bg/70 px-3 py-2">
          <div className="text-[10px] uppercase tracking-[0.14em] text-fg-disabled">Next action</div>
          <div className="mt-1 text-[12px] text-fg">{data.next_action}</div>
        </div>
        {tail.length ? (
          <div className="rounded-sm border border-border-subtle bg-bg-subtle/70">
            <div className="flex items-center justify-between border-b border-border-subtle px-3 py-2">
              <span className="text-[10px] uppercase tracking-[0.14em] text-fg-disabled">Last useful log lines</span>
              {data.log_path ? (
                <span className="max-w-[56ch] truncate font-mono text-[10px] text-fg-disabled" title={data.log_path}>
                  {data.log_path}
                </span>
              ) : null}
            </div>
            <div className="space-y-1 px-3 py-2">
              {tail.map((line, index) => (
                <div key={`${index}-${line}`} className="break-words font-mono text-[11px] leading-5 text-fg-muted">
                  {line}
                </div>
              ))}
            </div>
          </div>
        ) : null}
        <div className="flex flex-wrap gap-2">
          {data.retry_route ? (
            <Button asChild size="sm" variant="primary">
              <a href={data.retry_route}>Retry in Train</a>
            </Button>
          ) : null}
          <Button asChild size="sm" variant="ghost">
            <Link to="/diagnostics">Open Diagnostics</Link>
          </Button>
          {data.docs_url ? (
            <Button asChild size="sm" variant="ghost">
              <a href={data.docs_url}>Open docs</a>
            </Button>
          ) : null}
        </div>
      </CardContent>
    </Card>
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

function LossCard({
  cycles,
  livePoints,
  stage,
}: {
  cycles: CycleMetric[];
  livePoints: NonNullable<RunLive["metric_points"]>;
  stage: RunLive["stage"] | null;
}) {
  type LossChartPoint = {
    step: number;
    train_loss?: number | null;
    eval_loss?: number | null;
  };
  const chartData: LossChartPoint[] = livePoints.length
    ? livePoints.map((point) => ({
        step: point.step,
        train_loss: point.train_loss,
        eval_loss: point.eval_loss,
      }))
    : cycles.map((cycle) => ({
        step: cycle.cycle,
        train_loss: cycle.train_loss,
        eval_loss: cycle.eval_loss,
      }));
  const emptyState =
    stage?.message ??
    (stage?.key === "loading_dataset"
      ? "Loading dataset. Loss appears after the first optimizer step."
      : stage?.key === "downloading_model"
        ? "Resolving model files. Loss appears after training starts."
        : "Waiting for the first optimizer step.");
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
          data={chartData}
          xKey="step"
          series={series}
          height={200}
          yFormat={(v) => v.toFixed(2)}
          emptyState={emptyState}
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

/* -------------------------------------------------------------------------
 * Cost card — Track P2 / F-R. Energy + dollar estimate from wall-clock
 * × backend nominal-power. Renders an "estimate" badge when source is
 * "nominal" so users know it isn't a meter reading.
 * ----------------------------------------------------------------------- */

function CostCard({ cost }: { cost?: RunCost }) {
  if (!cost) return null;
  const isMeasured = cost.source === "measured";
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardEyebrow>COST</CardEyebrow>
          <CardTitle>Energy &amp; spend</CardTitle>
          <Zap className="h-3.5 w-3.5 text-fg-disabled" />
        </div>
        <Badge tone={isMeasured ? "success" : "neutral"} dot size="sm">
          {isMeasured ? "measured" : "estimate"}
        </Badge>
      </CardHeader>
      <CardContent className="p-0 divide-y divide-border-subtle text-[12.5px]">
        <CostRow
          label="Wall clock"
          value={fmtDuration(cost.duration_seconds)}
          mono
        />
        <CostRow
          label="Power draw"
          value={`${cost.power_watts_estimated.toFixed(0)} W`}
          mono
          icon={<Plug className="h-3 w-3 text-fg-disabled" />}
        />
        <CostRow
          label="Energy"
          value={`${cost.energy_kwh.toFixed(3)} kWh`}
          mono
        />
        <CostRow
          label={`Cost @ $${cost.cost_per_kwh.toFixed(2)}/kWh`}
          value={`$${cost.cost_usd.toFixed(2)}`}
          mono
          emphasis
        />
        <CostRow label="Backend" value={cost.backend} mono className="text-fg-muted" />
      </CardContent>
    </Card>
  );
}

function CostRow({
  label,
  value,
  mono,
  emphasis,
  icon,
  className,
}: {
  label: string;
  value: string;
  mono?: boolean;
  emphasis?: boolean;
  icon?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className="flex items-start justify-between gap-3 px-3.5 py-2">
      <span className="text-fg-muted flex items-center gap-1.5">
        {icon}
        {label}
      </span>
      <span
        className={cn(
          "text-right truncate max-w-[18ch]",
          mono ? "font-mono" : "",
          emphasis ? "text-accent text-[13px]" : "text-fg",
          className,
        )}
        title={value}
      >
        {value}
      </span>
    </div>
  );
}

/* -------------------------------------------------------------------------
 * Lineage card — Track F-Q. Shows parents/children + a "Mark as fork"
 * affordance so the operator can record relationships between runs
 * (different LR, different dataset, different rank, etc.) without
 * stepping outside the dashboard.
 * ----------------------------------------------------------------------- */

function LineageCard({ runId }: { runId: string }) {
  const queryClient = useQueryClient();
  const lineageQuery = useQuery({
    queryKey: ["run-lineage", runId],
    queryFn: () => api.getRunLineage(runId),
    refetchInterval: 30_000,
    refetchIntervalInBackground: false,
  });

  const recordMutation = useMutation({
    mutationFn: (payload: { parent_run_id: string; forked_at_cycle?: number | null; notes?: string | null }) =>
      api.recordRunFork(runId, payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["run-lineage", runId] });
    },
  });

  const removeMutation = useMutation({
    mutationFn: (parentId: string) => api.removeRunFork(runId, parentId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["run-lineage", runId] });
    },
  });

  function recordFork() {
    const parent = window.prompt(
      "Parent run_id (the run this one was forked from):",
    );
    if (!parent) return;
    const cycleStr = window.prompt(
      "Forked at cycle? (blank = unknown)",
      "",
    );
    const cycleNum = cycleStr && cycleStr.trim() ? parseInt(cycleStr, 10) : null;
    const notes = window.prompt(
      "What changed? (e.g. 'lr 5e-6 → 1e-6')",
      "",
    );
    recordMutation.mutate({
      parent_run_id: parent.trim(),
      forked_at_cycle: Number.isFinite(cycleNum as number) ? (cycleNum as number) : null,
      notes: notes?.trim() || null,
    });
  }

  const data = lineageQuery.data;
  const hasAny = !!data && (data.ancestors.length > 0 || data.descendants.length > 0);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardEyebrow>LINEAGE</CardEyebrow>
          <CardTitle>Forks</CardTitle>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={recordFork}
          disabled={recordMutation.isPending}
          title="Mark this run as forked from another"
        >
          Mark as fork
        </Button>
      </CardHeader>
      <CardContent className="text-[12px]">
        {lineageQuery.isLoading ? (
          <div className="text-fg-muted text-[11px]">Loading lineage…</div>
        ) : !hasAny ? (
          <div className="text-fg-muted text-[11px] max-w-[44ch]">
            No recorded lineage yet. Use{" "}
            <span className="font-mono text-fg-subtle">Mark as fork</span>{" "}
            to record this run's parent (e.g. "forked from prod-baseline at cycle 4 with lr halved").
          </div>
        ) : (
          <div className="space-y-2">
            {data!.ancestors.length ? (
              <LineageGroup
                title="Parents"
                edges={data!.ancestors.map((e) => ({
                  ...e,
                  related_id: e.parent_run_id ?? "",
                  removable: e.depth === 1,
                }))}
                onRemove={(parentId) => removeMutation.mutate(parentId)}
                removePending={removeMutation.isPending}
              />
            ) : null}
            {data!.descendants.length ? (
              <LineageGroup
                title="Children"
                edges={data!.descendants.map((e) => ({
                  ...e,
                  related_id: e.child_run_id ?? "",
                  removable: false,
                }))}
              />
            ) : null}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function LineageGroup({
  title,
  edges,
  onRemove,
  removePending,
}: {
  title: string;
  edges: Array<{
    related_id: string;
    forked_at_cycle: number | null;
    notes: string | null;
    depth: number;
    removable: boolean;
  }>;
  onRemove?: (id: string) => void;
  removePending?: boolean;
}) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-[0.12em] text-fg-disabled mb-1">
        {title}
      </div>
      <div className="space-y-1">
        {edges
          .slice()
          .sort((a, b) => a.depth - b.depth)
          .map((e) => (
            <div
              key={`${e.related_id}-${e.depth}`}
              className="flex items-start justify-between gap-2 text-[11px]"
            >
              <div className="flex-1 min-w-0">
                <Link
                  to="/runs/$runId"
                  params={{ runId: e.related_id }}
                  className="font-mono text-accent hover:underline truncate inline-block max-w-[24ch]"
                  title={e.related_id}
                >
                  {e.related_id.length > 24
                    ? `${e.related_id.slice(0, 21)}…`
                    : e.related_id}
                </Link>
                <span className="text-fg-disabled ml-1.5">depth {e.depth}</span>
                {typeof e.forked_at_cycle === "number" ? (
                  <span className="text-fg-muted ml-1.5">
                    @ cycle {e.forked_at_cycle}
                  </span>
                ) : null}
                {e.notes ? (
                  <div className="text-fg-muted mt-0.5 truncate" title={e.notes}>
                    {e.notes}
                  </div>
                ) : null}
              </div>
              {e.removable && onRemove ? (
                <button
                  type="button"
                  onClick={() => {
                    if (confirm("Remove this lineage edge?")) {
                      onRemove(e.related_id);
                    }
                  }}
                  disabled={removePending}
                  aria-label="Remove lineage"
                  className="shrink-0 text-fg-disabled hover:text-fg p-0.5"
                >
                  <X className="h-3 w-3" />
                </button>
              ) : null}
            </div>
          ))}
      </div>
    </div>
  );
}

function DatasetBindingsCard({ bindings }: { bindings: DatasetBinding[] }) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardEyebrow>DATA</CardEyebrow>
          <CardTitle>Dataset bindings</CardTitle>
          <Database className="h-3.5 w-3.5 text-fg-disabled" />
        </div>
        <Badge tone={bindings.length ? "success" : "neutral"} size="sm">{bindings.length}</Badge>
      </CardHeader>
      <CardContent className="p-0">
        {bindings.length ? (
          <div className="divide-y divide-border-subtle">
            {bindings.map((binding) => {
              const body = (
                <>
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[10px] font-medium uppercase tracking-[0.12em] text-fg-subtle">{binding.role}</span>
                    <span className="font-mono text-[10px] text-fg-muted">{binding.split}</span>
                  </div>
                  <div className="mt-1 truncate font-mono text-[10.5px] text-accent" title={binding.dataset_version_id}>{binding.dataset_name || binding.dataset_version_id}</div>
                  <div className="mt-0.5 truncate font-mono text-[9.5px] text-fg-disabled" title={binding.content_hash || undefined}>{binding.content_hash?.slice(0, 16) || "legacy identity"}{binding.artifact_hash ? ` · artifact ${binding.artifact_hash.slice(0, 12)}` : ""}</div>
                </>
              );
              return binding.dataset_id ? (
                <Link key={`${binding.role}-${binding.dataset_version_id}`} to="/datasets/$datasetId/versions/$versionId" params={{ datasetId: binding.dataset_id, versionId: binding.dataset_version_id }} search={{ split: binding.split }} className="block px-3.5 py-2.5 hover:bg-surface-hover/30">{body}</Link>
              ) : (
                <div key={`${binding.role}-${binding.dataset_version_id}`} className="px-3.5 py-2.5">{body}</div>
              );
            })}
          </div>
        ) : (
          <div className="px-3.5 py-4 text-[11px] leading-4 text-fg-muted">Legacy run: no immutable Dataset Lab identity was attached.</div>
        )}
      </CardContent>
    </Card>
  );
}

function EvaluationHistoryCard({ runId, items, loading }: { runId: string; items: Evaluation[]; loading: boolean }) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardEyebrow>EVAL</CardEyebrow>
          <CardTitle>Evaluation history</CardTitle>
          <FlaskConical className="h-3.5 w-3.5 text-fg-disabled" />
        </div>
        <Button variant="ghost" size="sm" asChild><Link to="/eval" search={{ runId, suite: undefined, evaluation: undefined }}>Open lab</Link></Button>
      </CardHeader>
      <CardContent className="p-0">
        {loading ? <div className="px-3.5 py-4 text-[11px] text-fg-muted">Loading evaluations…</div> : items.length ? (
          <div className="divide-y divide-border-subtle">
            {items.slice(0, 6).map((evaluation) => (
              <Link key={evaluation.id} to="/eval" search={{ runId, suite: undefined, evaluation: evaluation.id }} className="flex items-start justify-between gap-3 px-3.5 py-2.5 hover:bg-surface-hover/30">
                <div className="min-w-0"><div className="truncate font-mono text-[10.5px] text-accent">{evaluation.id}</div><div className="mt-0.5 truncate text-[10px] text-fg-subtle">{evaluation.suite_name || evaluation.suite_revision_id}</div></div>
                <div className="shrink-0 text-right"><Badge tone={evaluationTone(evaluation.status)} dot size="sm">{evaluation.status}</Badge><div className="mt-1 font-mono text-[10px] text-fg-muted">{formatEvalMetric(evaluation)}</div></div>
              </Link>
            ))}
          </div>
        ) : <div className="px-3.5 py-4 text-[11px] leading-4 text-fg-muted">No completed evaluation is attached. Training loss is not counted as evaluation evidence.</div>}
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
  const isRewarded = ["raft", "grpo", "vlm", "audio", "reasoning", "agentic"].includes(modality);

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
              {isRewarded ? <Th align="right">Avg reward</Th> : null}
              {isRewarded ? <Th align="right">Kept</Th> : null}
              {isRewarded ? <Th align="right">Success</Th> : null}
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
                {isRewarded ? (
                  <Td align="right" mono>
                    {fmt(c.avg_reward, 3)}
                  </Td>
                ) : null}
                {isRewarded ? (
                  <Td align="right" mono className="text-fg-muted">
                    {c.samples_kept != null && c.samples_seen != null
                      ? `${c.samples_kept}/${c.samples_seen}`
                      : "—"}
                  </Td>
                ) : null}
                {isRewarded ? (
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

function summaryForMethod(modality: string, data: RunDetail): string {
  const mode = modality.toLowerCase();
  if (mode === "sft") return "SFT tracks labeled-example loss and final artifact availability.";
  if (mode === "raft") return "RAFT tracks generated, verified, kept, and trained samples by cycle.";
  if (mode === "grpo") return "GRPO tracks verifier reward, group updates, and policy loss.";
  if (mode === "dpo" || mode === "orpo") return "Preference tuning tracks chosen/rejected pair loss and final adapter availability.";
  if (mode === "rm") return "Reward-model training tracks scorer quality and reward-margin signal.";
  if (mode === "vlm") return "VLM training tracks vision-language cycles, reward signal, and artifacts.";
  if (mode === "audio") return "Audio training tracks task-specific cycles, verifier signal, and artifacts.";
  if (mode === "reasoning") return "Reasoning training tracks math/data yield, cycles, and final artifact availability.";
  if (mode === "agentic") return "Agentic training tracks tool-call format quality, cycles, and artifacts.";
  return data.details?.final_model_available ? "Final artifact is available." : "Run artifacts will appear as training progresses.";
}

function cycleLikeMethod(modality: string): boolean {
  return ["raft", "grpo", "vlm", "audio", "reasoning", "agentic"].includes(modality.toLowerCase());
}

function fmtDuration(seconds: number | null | undefined): string {
  if (seconds == null || seconds <= 0) return "—";
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}

function artifactStateLabel(state: string | null | undefined): string {
  const value = String(state ?? "none").toLowerCase();
  if (value === "final_model") return "final model";
  if (value === "checkpoint") return "checkpoint";
  if (value === "failed") return "failed";
  return "pending";
}

function formatProgress(current: number | null | undefined, total: number | null | undefined): string {
  if (current == null || Number.isNaN(current)) return "—";
  if (total == null || Number.isNaN(total) || total <= 0) return String(current);
  return `${current}/${total}`;
}

function plainRunStatus(
  status: string | undefined,
  streamStatus: "idle" | "connecting" | "open" | "closed" | "error",
  streamError: string | null,
): string {
  if (streamError === "Remote token required.") return "auth needed";
  if (streamStatus === "connecting") return "connecting";
  if (streamStatus === "error") return "reconnecting";
  const normalized = String(status ?? "").toLowerCase();
  if (normalized === "completed") return "completed";
  if (normalized === "failed") return "failed";
  if (normalized === "stopped" || normalized === "cancelled" || normalized === "canceled") {
    return "cancelled";
  }
  if (isJobRunning(normalized)) return "running";
  if (!normalized && streamStatus === "closed") return "backend unreachable";
  return normalized || "unknown";
}

function isJobRunning(status: string | undefined): boolean {
  if (!status) return false;
  const s = status.toLowerCase();
  return s === "running" || s === "active" || s === "in_progress" || s === "pending";
}

function evaluationTone(status: string): "neutral" | "accent" | "success" | "warning" | "danger" {
  if (status === "completed") return "success";
  if (status === "failed") return "danger";
  if (status === "cancelled") return "warning";
  if (["queued", "running"].includes(status)) return "accent";
  return "neutral";
}

function formatEvalMetric(evaluation: Evaluation): string {
  const metric = evaluation.primary_metric ?? evaluation.metrics?.[0];
  if (!metric) return "—";
  return `${metric.name} ${typeof metric.value === "number" ? metric.value.toFixed(4) : "—"}`;
}
