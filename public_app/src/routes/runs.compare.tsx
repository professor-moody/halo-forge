import { createFileRoute, Link } from "@tanstack/react-router";
import { useQueries } from "@tanstack/react-query";
import {
  ArrowLeft,
  GitCompareArrows,
  Trash2,
  X,
} from "lucide-react";
import { useMemo } from "react";
import { api, type CycleMetric, type RunDetail } from "@/lib/api";
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
import {
  colorForIndex,
  MultiRunChart,
  type RunSeries,
} from "@/components/charts/multi-run-chart";
import { clearPinned, unpinRun, usePinnedRuns } from "@/lib/pinned-runs";
import { cn } from "@/lib/utils";

/**
 * Multi-run comparison view.
 *
 * Anatomy (top to bottom):
 *
 *   Topbar         "Comparing N runs" + clear all
 *   Run chips row  one chip per pinned run (id + color + unpin button)
 *   Loss overlay   train_loss across all runs, color-coded
 *   Reward overlay avg_reward across RAFT runs (skip if none)
 *   Config diff    side-by-side table of differing fields
 *
 * No new backend — fetches each run's detail via GET /runs/{id} in
 * parallel (Tanstack useQueries) and pivots the cycle_metrics arrays
 * client-side. This is fine for the 6-run cap; if we ever need
 * thousands of runs in a comparison we'd add a /runs/compare endpoint.
 */

export const Route = createFileRoute("/runs/compare")({
  component: CompareRoute,
});

function CompareRoute() {
  const ids = usePinnedRuns();

  const queries = useQueries({
    queries: ids.map((id) => ({
      queryKey: ["run-detail", id, "compare"],
      queryFn: () => api.runDetail(id),
      // Comparison is read-mostly; refetch on focus so a re-load picks
      // up new cycles for active runs without a route navigation.
      refetchInterval: 10_000,
      refetchIntervalInBackground: false,
    })),
  });

  const runs = useMemo(
    () =>
      queries
        .map((q, i) => ({
          id: ids[i],
          color: colorForIndex(i),
          status: q.status,
          data: q.data as RunDetail | undefined,
          error: q.error as Error | null | undefined,
        }))
        .filter((r) => Boolean(r.id)),
    [queries, ids],
  );

  const lossSeries = useMemo<RunSeries[]>(
    () =>
      runs
        .filter((r) => r.data)
        .map((r) => ({
          id: r.id,
          color: r.color,
          data: ((r.data!.details?.cycle_metrics ?? []) as CycleMetric[]).map(
            (c) => ({ cycle: c.cycle, value: c.train_loss }),
          ),
        })),
    [runs],
  );

  const rewardSeries = useMemo<RunSeries[]>(
    () =>
      runs
        .filter((r) => r.data && r.data.modality === "raft")
        .map((r) => ({
          id: r.id,
          color: r.color,
          data: ((r.data!.details?.cycle_metrics ?? []) as CycleMetric[]).map(
            (c) => ({ cycle: c.cycle, value: c.avg_kept_reward }),
          ),
        })),
    [runs],
  );

  return (
    <>
      <Topbar
        eyebrow="Workspace"
        title={ids.length === 0 ? "Compare runs" : `Comparing ${ids.length} runs`}
        subtitle={
          ids.length === 0
            ? "Pin runs from the run detail view to compare them here."
            : "Loss and reward curves overlaid across pinned runs."
        }
        actions={
          <>
            <Button variant="ghost" size="icon" asChild aria-label="Back to runs">
              <Link to="/runs">
                <ArrowLeft />
              </Link>
            </Button>
            {ids.length ? (
              <Button
                variant="ghost"
                size="sm"
                onClick={clearPinned}
                title="Unpin all runs"
              >
                <Trash2 />
                Clear all
              </Button>
            ) : null}
          </>
        }
      />

      {ids.length === 0 ? (
        <EmptyState />
      ) : (
        <div className="px-5 py-5 space-y-4">
          <RunChipsRow runs={runs} />
          <LossOverlay series={lossSeries} runs={runs} />
          {rewardSeries.length ? (
            <RewardOverlay series={rewardSeries} />
          ) : null}
          <ConfigDiff runs={runs} />
        </div>
      )}
    </>
  );
}

/* -------------------------------------------------------------------------
 * Empty state — when no runs are pinned, the route teaches you how to
 * fill the comparison tray.
 * ----------------------------------------------------------------------- */

function EmptyState() {
  return (
    <div className="px-5 py-12">
      <Card>
        <CardContent className="text-center space-y-3 py-12">
          <div className="mx-auto flex h-9 w-9 items-center justify-center rounded-md border border-border-subtle bg-surface">
            <GitCompareArrows className="h-4 w-4 text-fg-subtle" />
          </div>
          <div className="text-sm font-medium text-fg">No runs pinned</div>
          <div className="text-xs text-fg-muted max-w-[44ch] mx-auto">
            Open a run from the Runs list and click the <span className="font-mono text-fg">Pin</span>{" "}
            button. Pinned runs stack here so you can overlay their loss and
            reward curves to see how config changes moved the needle.
          </div>
          <Button asChild variant="primary" size="sm">
            <Link to="/runs">Browse runs</Link>
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}

/* -------------------------------------------------------------------------
 * Run chips — one per pinned run. Color matches the chart legend.
 * ----------------------------------------------------------------------- */

interface RunChip {
  id: string;
  color: string;
  status: "pending" | "loading" | "success" | "error" | string;
  data: RunDetail | undefined;
  error: Error | null | undefined;
}

function RunChipsRow({ runs }: { runs: RunChip[] }) {
  return (
    <div className="flex flex-wrap gap-2">
      {runs.map((run) => (
        <div
          key={run.id}
          className="flex items-center gap-2 rounded-md border border-border-subtle bg-surface pr-1 pl-2 py-1"
        >
          <span
            className="status-dot"
            style={{ background: run.color }}
            aria-hidden
          />
          <Link
            to="/runs/$runId"
            params={{ runId: run.id }}
            className="font-mono text-[11px] text-accent hover:underline"
          >
            {run.id.length > 24 ? `${run.id.slice(0, 21)}…` : run.id}
          </Link>
          {run.data ? (
            <span className="text-[10px] text-fg-subtle font-mono">
              {run.data.modality}
            </span>
          ) : null}
          {run.status === "error" ? (
            <Badge tone="danger" size="sm">
              error
            </Badge>
          ) : run.status === "pending" ? (
            <Badge tone="neutral" size="sm">
              loading
            </Badge>
          ) : null}
          <button
            type="button"
            onClick={() => unpinRun(run.id)}
            aria-label={`Unpin ${run.id}`}
            className="ml-0.5 h-5 w-5 rounded-sm text-fg-disabled hover:text-fg hover:bg-surface-hover transition-colors flex items-center justify-center"
          >
            <X className="h-3 w-3" />
          </button>
        </div>
      ))}
    </div>
  );
}

/* -------------------------------------------------------------------------
 * Charts
 * ----------------------------------------------------------------------- */

function LossOverlay({
  series,
  runs,
}: {
  series: RunSeries[];
  runs: RunChip[];
}) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardEyebrow>OVERLAY</CardEyebrow>
          <CardTitle>Train loss</CardTitle>
        </div>
        <span className="text-[11px] text-fg-subtle">
          {series.length} of {runs.length} loaded
        </span>
      </CardHeader>
      <CardContent className="px-3 py-4">
        <MultiRunChart
          series={series}
          format={(v) => v.toFixed(4)}
          height={240}
          emptyState="Waiting for run details to load…"
        />
      </CardContent>
    </Card>
  );
}

function RewardOverlay({ series }: { series: RunSeries[] }) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardEyebrow>RLVR</CardEyebrow>
          <CardTitle>Avg kept reward</CardTitle>
        </div>
        <span className="text-[11px] text-fg-subtle">
          RAFT runs only — {series.length} compared
        </span>
      </CardHeader>
      <CardContent className="px-3 py-4">
        <MultiRunChart
          series={series}
          format={(v) => v.toFixed(3)}
          height={200}
          emptyState="No RAFT runs in this comparison."
        />
      </CardContent>
    </Card>
  );
}

/* -------------------------------------------------------------------------
 * Config diff — side-by-side table of fields where the runs differ.
 * Aligned columns mean identical values blend together visually; rows
 * that don't fully agree get a faint copper underline so the eye lands
 * on them.
 * ----------------------------------------------------------------------- */

function ConfigDiff({ runs }: { runs: RunChip[] }) {
  const loaded = runs.filter((r) => r.data);
  if (loaded.length < 2) return null;

  const fields: Array<{ label: string; pick: (r: RunDetail) => string }> = [
    { label: "Modality", pick: (r) => String(r.modality ?? "—") },
    { label: "Model", pick: (r) => String(r.model_name ?? "—") },
    { label: "Status", pick: (r) => String(r.status ?? "—") },
    {
      label: "Cycles",
      pick: (r) => String(r.details?.cycles_executed ?? "—"),
    },
    {
      label: "Final loss",
      pick: (r) =>
        typeof r.metrics_summary?.final_train_loss === "number"
          ? r.metrics_summary.final_train_loss.toFixed(3)
          : "—",
    },
    {
      label: "Steps",
      pick: (r) => String(r.metrics_summary?.update_steps ?? "—"),
    },
    {
      label: "Verdict",
      pick: (r) => String(r.user_summary?.confidence_tone ?? "—"),
    },
    { label: "Seed", pick: (r) => String(r.details?.seed ?? "—") },
  ];

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardEyebrow>DIFF</CardEyebrow>
          <CardTitle>Config</CardTitle>
        </div>
        <span className="text-[11px] text-fg-subtle">
          {loaded.length} runs · differing fields highlighted
        </span>
      </CardHeader>
      <CardContent className="p-0 overflow-x-auto">
        <table className="w-full text-[12.5px] min-w-max">
          <thead>
            <tr className="border-b border-border-subtle">
              <th className="px-3.5 py-2 text-left text-[10px] font-medium uppercase tracking-[0.12em] text-fg-disabled sticky left-0 bg-surface">
                Field
              </th>
              {loaded.map((r) => (
                <th
                  key={r.id}
                  className="px-3.5 py-2 text-left text-[10px] font-medium uppercase tracking-[0.12em] text-fg-disabled"
                >
                  <span className="flex items-center gap-1.5">
                    <span
                      className="status-dot"
                      style={{ background: r.color }}
                    />
                    <span className="font-mono normal-case tracking-normal text-fg-muted truncate max-w-[14ch]">
                      {r.id.length > 18 ? `${r.id.slice(0, 15)}…` : r.id}
                    </span>
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {fields.map((f) => {
              const values = loaded.map((r) => f.pick(r.data!));
              const allSame = values.every((v) => v === values[0]);
              return (
                <tr
                  key={f.label}
                  className={cn(
                    "border-b border-border-subtle last:border-0",
                    !allSame && "bg-accent-bg/20",
                  )}
                >
                  <td className="px-3.5 py-2 text-fg-muted sticky left-0 bg-surface whitespace-nowrap">
                    {f.label}
                  </td>
                  {values.map((v, i) => (
                    <td
                      key={`${f.label}-${i}`}
                      className={cn(
                        "px-3.5 py-2 font-mono text-[12px]",
                        !allSame ? "text-fg" : "text-fg-muted",
                      )}
                    >
                      <span className="truncate inline-block max-w-[24ch]" title={v}>
                        {v}
                      </span>
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}
