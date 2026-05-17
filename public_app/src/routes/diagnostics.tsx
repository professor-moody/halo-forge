import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  FileText,
  Loader2,
  RefreshCw,
  Terminal,
} from "lucide-react";
import { useState } from "react";
import {
  api,
  type DiagnosticsLaunch,
  type DiagnosticsLogFile,
} from "@/lib/api";
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
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { cn, relativeTime } from "@/lib/utils";

export const Route = createFileRoute("/diagnostics")({
  component: DiagnosticsRoute,
});

/**
 * Diagnostics. Three things in one place:
 *
 *   1. Orphan launches — directories with launch_context.json but no
 *      training_summary.json. These are the runs that fired and died
 *      before producing the summary the /runs page reads from. Inline
 *      log tail per orphan so the failure is one click away.
 *   2. Completed launches — same source, status="completed". For
 *      cross-reference; the main view is /runs.
 *   3. logs/ inventory — every *.log under the project's logs/ dir.
 *      Click to tail. Useful for app-level logs that aren't tied to a
 *      specific run (uvicorn errors, cli misuses, etc.).
 */

function DiagnosticsRoute() {
  const summaryQuery = useQuery({
    queryKey: ["diagnostics", "summary"],
    queryFn: () => api.diagnosticsSummary(),
    refetchInterval: 10_000,
  });
  const launchesQuery = useQuery({
    queryKey: ["diagnostics", "launches"],
    queryFn: () => api.diagnosticsLaunches(),
    refetchInterval: 15_000,
  });
  const logsQuery = useQuery({
    queryKey: ["diagnostics", "logs"],
    queryFn: () => api.diagnosticsLogs(),
    refetchInterval: 15_000,
  });

  const summary = summaryQuery.data;
  const launches = launchesQuery.data?.items ?? [];
  const logs = logsQuery.data?.items ?? [];

  const orphans = launches.filter((e) => e.status === "orphan");
  const completed = launches.filter((e) => e.status === "completed");

  function refetchAll() {
    summaryQuery.refetch();
    launchesQuery.refetch();
    logsQuery.refetch();
  }

  return (
    <>
      <Topbar
        eyebrow="Operator"
        title="Diagnostics"
        subtitle={
          summary
            ? `${summary.launches.orphan} orphan · ${summary.launches.completed} completed · ${summary.logs.total} logs · ${summary.base_path}`
            : "Loading…"
        }
        actions={
          <Button variant="ghost" size="sm" onClick={refetchAll}>
            <RefreshCw className="h-3.5 w-3.5" /> Refresh
          </Button>
        }
      />

      <div className="px-5 py-5 space-y-6 max-w-5xl">
        {/* Orphan launches — top of the page because this is the most
            common reason a user asks "where's my run". */}
        <section aria-labelledby="orphans-h">
          <h2
            id="orphans-h"
            className="mb-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-fg-muted flex items-center gap-1.5"
          >
            <AlertTriangle className="h-3 w-3 text-warning" />
            Failed / orphan launches
            <span className="ml-1 text-fg-disabled font-normal normal-case tracking-normal">
              {orphans.length === 0
                ? "(none — all clear)"
                : "(no training_summary.json yet — likely failed before completion)"}
            </span>
          </h2>
          {launchesQuery.isLoading ? (
            <SkeletonRow />
          ) : orphans.length === 0 ? (
            <Card>
              <CardContent className="py-4 text-fg-muted text-[12px] flex items-center gap-2">
                <CheckCircle2 className="h-3.5 w-3.5 text-success" />
                No orphan launches under this directory.
              </CardContent>
            </Card>
          ) : (
            <ul className="space-y-2">
              {orphans.map((entry) => (
                <li key={entry.output_dir}>
                  <LaunchCard entry={entry} defaultOpen />
                </li>
              ))}
            </ul>
          )}
        </section>

        {/* Application logs — second so users debugging a non-run-tied
            issue don't have to scroll past every completed run. */}
        <section aria-labelledby="logs-h">
          <h2
            id="logs-h"
            className="mb-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-fg-muted flex items-center gap-1.5"
          >
            <Terminal className="h-3 w-3" />
            Application logs
            <span className="ml-1 text-fg-disabled font-normal normal-case tracking-normal">
              ({logs.length} files in logs/)
            </span>
          </h2>
          {logsQuery.isLoading ? (
            <SkeletonRow />
          ) : logs.length === 0 ? (
            <Card>
              <CardContent className="py-4 text-fg-muted text-[12px]">
                No Halo Forge application logs found for this workstation.
              </CardContent>
            </Card>
          ) : (
            <ul className="space-y-1.5">
              {logs.slice(0, 12).map((entry) => (
                <li key={entry.path}>
                  <LogFileRow entry={entry} />
                </li>
              ))}
              {logs.length > 12 ? (
                <li className="text-fg-disabled text-[11px] pt-1">
                  +{logs.length - 12} older log files
                </li>
              ) : null}
            </ul>
          )}
        </section>

        {/* Completed launches — collapsed by default, just for cross-ref. */}
        <section aria-labelledby="completed-h">
          <Collapsible>
            <CollapsibleTrigger asChild>
              <button className="flex w-full items-center gap-1.5 text-[11px] font-semibold uppercase tracking-[0.12em] text-fg-muted hover:text-fg transition-colors group">
                <ChevronDown className="h-3 w-3 transition-transform group-data-[state=closed]:-rotate-90" />
                <FileText className="h-3 w-3" />
                <span id="completed-h">Completed launches</span>
                <span className="ml-1 text-fg-disabled font-normal normal-case tracking-normal">
                  ({completed.length} — these also appear in /runs)
                </span>
              </button>
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-2">
              {completed.length === 0 ? (
                <Card>
                  <CardContent className="py-4 text-fg-muted text-[12px]">
                    No completed launches yet.
                  </CardContent>
                </Card>
              ) : (
                <ul className="space-y-2">
                  {completed.map((entry) => (
                    <li key={entry.output_dir}>
                      <LaunchCard entry={entry} />
                    </li>
                  ))}
                </ul>
              )}
            </CollapsibleContent>
          </Collapsible>
        </section>
      </div>
    </>
  );
}

function LaunchCard({
  entry,
  defaultOpen = false,
}: {
  entry: DiagnosticsLaunch;
  defaultOpen?: boolean;
}) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between gap-3">
          <div className="min-w-0 flex-1">
            <CardEyebrow>{entry.launched_at ? relativeTime(entry.launched_at) : "—"}</CardEyebrow>
            <CardTitle className="font-mono text-[13px] truncate">
              {entry.output_dir}
            </CardTitle>
          </div>
          {entry.status === "orphan" ? (
            <Badge tone="warning" size="sm">orphan</Badge>
          ) : (
            <Badge tone="success" size="sm">completed</Badge>
          )}
        </div>
      </CardHeader>
      <CardContent className="pt-0 space-y-2">
        {entry.command ? (
          <code className="block font-mono text-[11px] text-fg-disabled break-all">
            {entry.command.slice(2).join(" ")}
          </code>
        ) : null}
        {entry.log_files.length > 0 ? (
          <LogTailDisclosure path={entry.log_files[0]} defaultOpen={defaultOpen} />
        ) : (
          <div className="text-fg-disabled text-[11px]">No log files in run dir.</div>
        )}
      </CardContent>
    </Card>
  );
}

function LogFileRow({ entry }: { entry: DiagnosticsLogFile }) {
  return (
    <Card className="hover:border-border-strong transition-colors">
      <CardContent className="py-2.5 flex items-center gap-3">
        <FileText className="h-3.5 w-3.5 text-fg-muted shrink-0" />
        <span className="font-mono text-[12px] text-fg flex-1 truncate">
          {entry.name}
        </span>
        <span className="font-mono text-[11px] text-fg-disabled shrink-0">
          {formatBytes(entry.size_bytes)}
        </span>
        <span className="font-mono text-[11px] text-fg-disabled shrink-0">
          {relativeTime(new Date(entry.mtime * 1000).toISOString())}
        </span>
      </CardContent>
      <div className="px-3 pb-2">
        <LogTailDisclosure path={entry.path} />
      </div>
    </Card>
  );
}

function LogTailDisclosure({
  path,
  defaultOpen = false,
}: {
  path: string;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger asChild>
        <button
          className={cn(
            "flex items-center gap-1.5 text-[11px] font-medium transition-colors",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent rounded-sm",
            open ? "text-accent" : "text-fg-muted hover:text-fg",
          )}
        >
          <ChevronDown
            className={cn("h-3 w-3 transition-transform", open ? "" : "-rotate-90")}
          />
          {open ? "Hide log tail" : "Show log tail"}
          <span className="font-mono text-fg-disabled font-normal">
            {path.split("/").slice(-1)[0]}
          </span>
        </button>
      </CollapsibleTrigger>
      <CollapsibleContent>
        {open ? <LogTailBody path={path} /> : null}
      </CollapsibleContent>
    </Collapsible>
  );
}

function LogTailBody({ path }: { path: string }) {
  const { data, isLoading, isError, refetch } = useQuery({
    queryKey: ["diagnostics", "log-tail", path],
    queryFn: () => api.diagnosticsLogTail(path, 200),
    refetchInterval: 5_000,
  });

  if (isLoading) {
    return (
      <div className="mt-2 h-24 flex items-center justify-center text-fg-muted text-[11px] gap-1.5">
        <Loader2 className="h-3 w-3 animate-spin" /> Tailing…
      </div>
    );
  }
  if (isError || !data) {
    return (
      <div className="mt-2 text-danger text-[11px]">
        Failed to tail log.
        <button
          onClick={() => refetch()}
          className="ml-2 underline hover:text-danger/80"
        >
          retry
        </button>
      </div>
    );
  }
  if (!data.available) {
    return (
      <div className="mt-2 text-fg-muted text-[11px]">
        {data.reason ?? "Log unavailable."}
      </div>
    );
  }

  return (
    <pre
      className={cn(
        "mt-2 max-h-72 overflow-auto rounded-md border border-border-subtle bg-bg-subtle px-3 py-2",
        "font-mono text-[11px] leading-snug text-fg whitespace-pre-wrap break-all",
      )}
      role="log"
      aria-live="polite"
    >
      {data.lines.length === 0 ? (
        <span className="text-fg-disabled">(empty)</span>
      ) : (
        data.lines.join("\n")
      )}
      {data.truncated_head ? (
        <div className="mt-2 text-fg-disabled text-[10px]">
          ⤴ Older lines truncated (file is {formatBytes(data.size_bytes ?? 0)}).
        </div>
      ) : null}
    </pre>
  );
}

function SkeletonRow() {
  return (
    <Card>
      <CardContent className="py-4 text-fg-muted text-[12px] flex items-center gap-2">
        <Loader2 className="h-3.5 w-3.5 animate-spin" /> Loading…
      </CardContent>
    </Card>
  );
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}
