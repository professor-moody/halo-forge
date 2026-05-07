import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useMutation, useQuery } from "@tanstack/react-query";
import { ArrowLeft, BarChart3, BookmarkPlus } from "lucide-react";
import { useMemo } from "react";
import { api, type EvalCohortResponse } from "@/lib/api";
import { Topbar } from "@/components/shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardEyebrow, CardHeader, CardTitle } from "@/components/ui/card";
import { usePinnedRuns } from "@/lib/pinned-runs";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/eval")({
  component: EvalCohortRoute,
});

/**
 * Cohort eval dashboard (Track F-K).
 *
 * Pulls every pinned run's `lm_eval_summary.json` into a runs × tasks
 * grid. Best-per-task is highlighted in copper. Runs without an
 * `lm_eval_summary.json` show up with a "no eval" reason; the user
 * can fix that by running `halo-forge eval --output <run_dir>` and
 * coming back.
 *
 * Why no run picker yet: the pinned-runs tray is already the
 * comparison surface; adding a second selector here would split
 * the workflow. If users want a different cohort they can pin/unpin
 * from the runs list and reload — same UX as /runs/compare.
 */

function EvalCohortRoute() {
  const pinned = usePinnedRuns();
  const navigate = useNavigate();

  const { data, isLoading, isError, error } = useQuery<EvalCohortResponse>({
    queryKey: ["eval-cohort", pinned],
    queryFn: () => api.evalCohort(pinned),
    enabled: pinned.length > 0,
    refetchInterval: 30_000,
    refetchIntervalInBackground: false,
  });

  const saveMutation = useMutation({
    mutationFn: api.createRegistryEntry,
    onSuccess: () => {
      navigate({ to: "/registry" });
    },
  });

  function handleSave() {
    const name = window.prompt(
      "Bundle name? (e.g. 'prod-2026-q2')",
      `cohort-${new Date().toISOString().slice(0, 10)}`,
    );
    if (!name) return;
    saveMutation.mutate({
      name: name.trim(),
      run_ids: pinned,
      tags: ["cohort"],
    });
  }

  return (
    <>
      <Topbar
        eyebrow="Workspace"
        title="Cohort eval"
        subtitle={
          pinned.length === 0
            ? "Pin runs from the runs list to compare their eval results."
            : `${pinned.length} runs · ${data?.tasks.length ?? 0} tasks`
        }
        actions={
          <>
            {pinned.length > 0 ? (
              <Button
                variant="ghost"
                size="sm"
                onClick={handleSave}
                disabled={saveMutation.isPending}
                title="Save these pinned runs as a named bundle"
              >
                <BookmarkPlus />
                Save selection
              </Button>
            ) : null}
            <Button variant="ghost" size="icon" asChild aria-label="Back to runs">
              <Link to="/runs">
                <ArrowLeft />
              </Link>
            </Button>
          </>
        }
      />

      {pinned.length === 0 ? (
        <EmptyState />
      ) : (
        <div className="px-5 py-5 space-y-4">
          {isLoading ? (
            <Card>
              <CardContent className="px-6 py-10 text-center text-sm text-fg-muted">
                Loading eval summaries…
              </CardContent>
            </Card>
          ) : isError ? (
            <Card>
              <CardContent className="px-6 py-8 text-center text-sm text-fg-muted">
                Cohort fetch failed: {(error as Error)?.message ?? "unknown error"}
              </CardContent>
            </Card>
          ) : data ? (
            <CohortTable data={data} />
          ) : null}
        </div>
      )}
    </>
  );
}

/* -------------------------------------------------------------------------
 * Empty state
 * ----------------------------------------------------------------------- */

function EmptyState() {
  return (
    <div className="px-5 py-12">
      <Card>
        <CardContent className="text-center space-y-3 py-12">
          <div className="mx-auto flex h-9 w-9 items-center justify-center rounded-md border border-border-subtle bg-surface">
            <BarChart3 className="h-4 w-4 text-fg-subtle" />
          </div>
          <div className="text-sm font-medium text-fg">No runs pinned</div>
          <div className="text-xs text-fg-muted max-w-[44ch] mx-auto">
            Pin runs from the runs list, then come back to see their eval
            summaries side by side. Run{" "}
            <span className="font-mono text-fg">halo-forge eval --output &lt;run_dir&gt;</span>{" "}
            on each run first to populate the eval data.
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
 * Cohort table
 * ----------------------------------------------------------------------- */

function CohortTable({ data }: { data: EvalCohortResponse }) {
  const { runs, tasks, cells, best_per_task_higher_is_better } = data;

  const availableCount = useMemo(
    () => runs.filter((r) => r.available).length,
    [runs],
  );

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardEyebrow>EVAL</CardEyebrow>
          <CardTitle>Cohort comparison</CardTitle>
          <BarChart3 className="h-3.5 w-3.5 text-fg-disabled" />
        </div>
        <span className="text-[11px] text-fg-subtle">
          {availableCount} of {runs.length} runs have eval summaries
        </span>
      </CardHeader>
      <CardContent className="p-0 overflow-x-auto">
        <table className="w-full text-[12.5px] min-w-max">
          <thead>
            <tr className="border-b border-border-subtle">
              <th className="px-3.5 py-2 text-left text-[10px] font-medium uppercase tracking-[0.12em] text-fg-disabled sticky left-0 bg-surface">
                Run
              </th>
              <th className="px-3.5 py-2 text-left text-[10px] font-medium uppercase tracking-[0.12em] text-fg-disabled">
                Status
              </th>
              {tasks.map((task) => (
                <th
                  key={task}
                  className="px-3.5 py-2 text-right text-[10px] font-medium uppercase tracking-[0.12em] text-fg-disabled"
                >
                  {task}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => {
              const cellsForRun = cells[run.run_id] ?? {};
              return (
                <tr
                  key={run.run_id}
                  className="border-b border-border-subtle last:border-0 hover:bg-surface-hover/40 transition-colors"
                >
                  <td className="px-3.5 py-2 sticky left-0 bg-surface">
                    <Link
                      to="/runs/$runId"
                      params={{ runId: run.run_id }}
                      className="font-mono text-[11px] text-accent hover:underline"
                      title={run.model_name ?? run.run_id}
                    >
                      {run.run_id.length > 22
                        ? `${run.run_id.slice(0, 19)}…`
                        : run.run_id}
                    </Link>
                  </td>
                  <td className="px-3.5 py-2">
                    {run.available ? (
                      <Badge tone="success" dot size="sm">
                        ready
                      </Badge>
                    ) : (
                      <span
                        className="text-[11px] text-fg-disabled"
                        title={run.reason ?? "no eval summary"}
                      >
                        no eval
                      </span>
                    )}
                  </td>
                  {tasks.map((task) => {
                    const cell = cellsForRun[task];
                    const isBest =
                      best_per_task_higher_is_better[task] === run.run_id;
                    return (
                      <td
                        key={`${run.run_id}-${task}`}
                        className={cn(
                          "px-3.5 py-2 text-right font-mono tabular-nums",
                          isBest ? "text-accent font-medium" : "text-fg",
                          !cell || cell.error ? "text-fg-disabled" : null,
                        )}
                      >
                        {cell && typeof cell.value === "number"
                          ? cell.value.toFixed(4)
                          : "—"}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}
