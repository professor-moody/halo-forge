import { createFileRoute, Link } from "@tanstack/react-router";
import { useRuns } from "@/lib/hooks";
import { Topbar } from "@/components/shell";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { relativeTime } from "@/lib/utils";

export const Route = createFileRoute("/runs/")({
  component: RunsListRoute,
});

function RunsListRoute() {
  const { data, isLoading } = useRuns({ limit: 50 });
  const items = data?.items ?? [];

  return (
    <>
      <Topbar eyebrow="Workspace" title="Runs" subtitle="All training runs on this host." />
      <div className="px-6 py-6">
        <Card>
          <CardContent className="p-0">
            {isLoading ? (
              <div className="space-y-px p-6">
                {[0, 1, 2, 3, 4].map((i) => (
                  <div key={i} className="h-12 animate-pulse bg-surface-hover/40" />
                ))}
              </div>
            ) : items.length === 0 ? (
              <div className="px-6 py-12 text-center text-sm text-fg-muted">
                No runs found.
              </div>
            ) : (
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border-subtle text-[11px] uppercase tracking-wider text-fg-subtle">
                    <th className="px-4 py-2 text-left font-medium">Run</th>
                    <th className="px-4 py-2 text-left font-medium">Modality</th>
                    <th className="px-4 py-2 text-left font-medium">Model</th>
                    <th className="px-4 py-2 text-left font-medium">Status</th>
                    <th className="px-4 py-2 text-right font-medium">Cycles</th>
                    <th className="px-4 py-2 text-right font-medium">Loss</th>
                    <th className="px-4 py-2 text-right font-medium">When</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((run) => {
                    const verdict = run.effectiveness?.verdict;
                    const tone =
                      verdict === "passed"
                        ? "success"
                        : verdict === "failed"
                          ? "danger"
                          : verdict
                            ? "warning"
                            : "neutral";
                    return (
                      <tr
                        key={run.run_id}
                        className="border-b border-border-subtle last:border-0 hover:bg-surface-hover transition-colors"
                      >
                        <td className="px-4 py-2.5">
                          <Link
                            to="/runs/$runId"
                            params={{ runId: run.run_id }}
                            className="font-mono text-xs text-accent hover:underline"
                          >
                            {run.run_id.slice(0, 16)}
                          </Link>
                        </td>
                        <td className="px-4 py-2.5 capitalize text-fg-muted">{run.modality}</td>
                        <td className="px-4 py-2.5 truncate max-w-[28ch] text-fg">
                          {run.model_name}
                        </td>
                        <td className="px-4 py-2.5">
                          <Badge tone={tone} dot size="sm">
                            {verdict ?? "pending"}
                          </Badge>
                        </td>
                        <td className="px-4 py-2.5 text-right font-mono text-fg-muted">
                          {run.cycles_executed ?? "—"}
                        </td>
                        <td className="px-4 py-2.5 text-right font-mono text-fg">
                          {typeof run.final_train_loss === "number"
                            ? run.final_train_loss.toFixed(3)
                            : "—"}
                        </td>
                        <td className="px-4 py-2.5 text-right text-fg-muted whitespace-nowrap">
                          {relativeTime(run.created_at)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </CardContent>
        </Card>
      </div>
    </>
  );
}
