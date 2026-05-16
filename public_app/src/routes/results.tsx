import { createFileRoute, Link } from "@tanstack/react-router";
import { useQueryClient } from "@tanstack/react-query";
import {
  ArrowUpRight,
  CheckCircle2,
  FileText,
  GitCompareArrows,
  Loader2,
  ScrollText,
  Server,
} from "lucide-react";
import { Topbar } from "@/components/shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardEyebrow } from "@/components/ui/card";
import { queryKeys, useRunSearch, useServeStart, useServeStatus } from "@/lib/hooks";
import type { RunListItem } from "@/lib/api";
import { cn, relativeTime } from "@/lib/utils";

export const Route = createFileRoute("/results")({
  component: ResultsRoute,
});

function ResultsRoute() {
  const runs = useRunSearch({
    status: ["completed"],
    sortBy: "timestamp",
    sortDir: "desc",
    limit: 100,
  });
  const items = runs.data?.items ?? [];
  const servedArtifacts = items.filter((run) => run.final_model_available).length;
  const bestLoss = items
    .map((run) => (typeof run.final_train_loss === "number" ? run.final_train_loss : null))
    .filter((loss): loss is number => loss != null)
    .sort((a, b) => a - b)[0];

  return (
    <>
      <Topbar
        eyebrow="Workspace"
        title="Results"
        subtitle="Completed runs, artifacts, and next actions."
        actions={
          <Button asChild variant="ghost" size="sm">
            <Link to="/runs">
              All runs <ArrowUpRight />
            </Link>
          </Button>
        }
      />
      <div className="px-5 py-5 space-y-4">
        <div className="grid gap-2 md:grid-cols-3">
          <SummaryTile label="Completed" value={String(items.length)} />
          <SummaryTile label="Serve-ready" value={String(servedArtifacts)} />
          <SummaryTile label="Best loss" value={bestLoss != null ? bestLoss.toFixed(4) : "-"} />
        </div>

        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <CardEyebrow>COMPLETED</CardEyebrow>
              <CardTitle>Run results</CardTitle>
            </div>
            <span className="text-[11px] text-fg-subtle">
              {runs.isFetching ? "Refreshing" : `${items.length} indexed`}
            </span>
          </CardHeader>
          <CardContent className="p-0">
            {runs.isLoading ? (
              <div className="flex h-32 items-center justify-center gap-2 text-sm text-fg-muted">
                <Loader2 className="h-4 w-4 animate-spin" />
                Loading completed runs...
              </div>
            ) : runs.isError ? (
              <div className="px-5 py-10 text-sm text-danger">
                Failed to load completed results.
              </div>
            ) : items.length === 0 ? (
              <EmptyResults />
            ) : (
              <div className="divide-y divide-border-subtle">
                {items.map((run) => (
                  <ResultRow key={run.run_id} run={run} />
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </>
  );
}

function SummaryTile({ label, value }: { label: string; value: string }) {
  return (
    <Card>
      <CardContent className="px-3.5 py-3">
        <div className="text-[10px] font-medium uppercase tracking-[0.14em] text-fg-disabled">
          {label}
        </div>
        <div className="mt-1 font-mono text-[22px] leading-none text-fg">{value}</div>
      </CardContent>
    </Card>
  );
}

function ResultRow({ run }: { run: RunListItem }) {
  const queryClient = useQueryClient();
  const serveStatus = useServeStatus();
  const serveStart = useServeStart();
  const artifact = typeof run.artifact_path === "string" ? run.artifact_path : null;
  const outputDir = typeof run.output_dir === "string" ? run.output_dir : null;
  const loss =
    typeof run.final_train_loss === "number" ? run.final_train_loss.toFixed(4) : "-";
  const verdict = run.effectiveness?.verdict ?? "completed";
  const servingThis = Boolean(artifact && serveStatus.data?.running && serveStatus.data.model === artifact);
  const anotherModelServing = Boolean(serveStatus.data?.running && !servingThis);
  const serveDisabled =
    !artifact || serveStart.isPending || Boolean(serveStatus.data?.running && !servingThis);

  return (
    <div className="grid gap-3 px-4 py-3 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-center">
      <div className="min-w-0 space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <CheckCircle2 className="h-4 w-4 text-success" />
          <Link
            to="/runs/$runId"
            params={{ runId: run.run_id }}
            className="font-mono text-[13px] text-accent hover:underline"
          >
            {run.run_id}
          </Link>
          <Badge tone="success" dot size="sm">
            {verdict}
          </Badge>
          {artifact ? (
            <Badge tone="info" size="sm">
              artifact
            </Badge>
          ) : (
            <Badge tone="neutral" size="sm">
              no final model
            </Badge>
          )}
        </div>
        <div className="grid gap-2 text-[12px] text-fg-muted md:grid-cols-4">
          <Readout label="Model" value={run.model_name || "-"} mono />
          <Readout label="Modality" value={run.modality || "-"} />
          <Readout label="Loss" value={loss} mono />
          <Readout label="When" value={relativeTime(run.created_at ?? run.timestamp)} />
        </div>
        <div className="flex min-w-0 flex-wrap items-center gap-2 text-[11px] text-fg-disabled">
          <FileText className="h-3.5 w-3.5" />
          <span>Local workstation path:</span>
          <span className={cn("truncate font-mono", outputDir ? "text-fg-subtle" : "text-fg-disabled")}>
            {artifact ?? outputDir ?? "No output path indexed"}
          </span>
        </div>
        {anotherModelServing ? (
          <div className="rounded-sm border border-warning/30 bg-warning-bg px-2 py-1.5 text-[11px] text-warning">
            {serveStatus.data?.model} is already serving. Stop it in Playground before serving this result.
          </div>
        ) : null}
        {serveStart.error ? (
          <div className="rounded-sm border border-danger/30 bg-danger-bg px-2 py-1.5 text-[11px] text-danger">
            {(serveStart.error as Error).message}
          </div>
        ) : null}
      </div>
      <div className="flex flex-wrap gap-2 lg:justify-end">
        <Button asChild size="sm" variant="primary">
          <Link to="/runs/$runId" params={{ runId: run.run_id }}>
            Open run
          </Link>
        </Button>
        <Button asChild size="sm" variant="ghost">
          <Link to="/runs/$runId" params={{ runId: run.run_id }}>
            <ScrollText className="h-3.5 w-3.5" />
            View logs
          </Link>
        </Button>
        <Button
          size="sm"
          variant="ghost"
          disabled={!outputDir}
          title={outputDir ? outputDir : "No result files were indexed for this run."}
        >
          <FileText className="h-3.5 w-3.5" />
          Results files
        </Button>
        <Button
          size="sm"
          variant="ghost"
          disabled={serveDisabled}
          onClick={() =>
            artifact
              ? serveStart.mutate(
                  { model: artifact },
                  { onSettled: () => queryClient.invalidateQueries({ queryKey: queryKeys.serve }) },
                )
              : undefined
          }
          title={
            artifact
              ? "Serve this final model locally"
              : "No final model artifact was indexed for this run."
          }
        >
          <Server className="h-3.5 w-3.5" />
          {servingThis ? "Serving" : "Serve model"}
        </Button>
        {servingThis ? (
          <Button asChild size="sm" variant="primary">
            <Link to="/playground">Open Playground</Link>
          </Button>
        ) : null}
        <Button asChild size="sm" variant="ghost">
          <Link to="/runs/compare">
            <GitCompareArrows className="h-3.5 w-3.5" />
            Compare
          </Link>
        </Button>
      </div>
    </div>
  );
}

function Readout({
  label,
  value,
  mono,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="min-w-0">
      <div className="text-[10px] uppercase tracking-[0.12em] text-fg-disabled">{label}</div>
      <div className={cn("truncate text-fg", mono && "font-mono")}>{value}</div>
    </div>
  );
}

function EmptyResults() {
  return (
    <div className="flex flex-col items-center justify-center px-6 py-12 text-center">
      <div className="flex h-9 w-9 items-center justify-center rounded-md border border-border-subtle bg-surface">
        <CheckCircle2 className="h-4 w-4 text-fg-subtle" />
      </div>
      <div className="mt-3 text-[13px] font-medium text-fg">No completed results yet</div>
      <div className="mt-1 max-w-[38ch] text-xs text-fg-muted">
        Completed runs with metrics and artifacts will appear here.
      </div>
      <Button asChild variant="primary" size="sm" className="mt-3.5">
        <Link to="/start" search={{ goal: undefined }}>Start run</Link>
      </Button>
    </div>
  );
}
