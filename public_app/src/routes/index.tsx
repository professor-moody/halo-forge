import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import {
  Activity,
  AlertTriangle,
  ArrowUpRight,
  CheckCircle2,
  CircleDashed,
  Clock3,
  Cpu,
  Plus,
  RefreshCw,
  Zap,
} from "lucide-react";
import { useActivity, useDashboard, useRuns, useBackendInfo } from "@/lib/hooks";
import { api, type ActivityItem, type ResearchDecisionRecord } from "@/lib/api";
import { Topbar } from "@/components/shell";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { compactNumber, relativeTime, cn } from "@/lib/utils";
import type { RunListItem } from "@/lib/api";

/**
 * Overview — the design exemplar. Anatomy, top-to-bottom:
 *
 *   Topbar                page identity + last-updated mono readout
 *   Telemetry strip       hardware vitals (placeholder until Phase B)
 *   Stat ribbon           four mono-numeric tiles, monospace dominant
 *   Two-column            recent-runs table (2/3) | system summary (1/3)
 *
 * Density rules: 8px vertical rhythm at the page edge, 12px inside cards,
 * 4px between rows. Mono used for any value an operator might compare
 * across rows (run id, loss, cycles, throughput, dtype, attn impl).
 */

export const Route = createFileRoute("/")({
  component: OverviewRoute,
});

function OverviewRoute() {
  const dashboard = useDashboard();
  const runs = useRuns({ limit: 8 });
  const backend = useBackendInfo();
  const activity = useActivity(100);
  const decisions = useQuery({ queryKey: ["research-decisions", "overview"], queryFn: () => api.listResearchDecisions({ limit: 6 }), retry: false });
  const readiness = useQuery({ queryKey: ["workstation-readiness"], queryFn: () => api.workstationReadiness(), retry: false });

  const lastRefreshed = dashboard.dataUpdatedAt
    ? relativeTime(dashboard.dataUpdatedAt)
    : "—";

  const items = runs.data?.items ?? [];

  return (
    <>
      <Topbar
        eyebrow="Workspace"
        title="Overview"
        actions={
          <>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => {
                dashboard.refetch();
                runs.refetch();
              }}
              disabled={dashboard.isFetching}
              aria-label="Refresh"
            >
              <RefreshCw className={dashboard.isFetching ? "animate-spin" : undefined} />
            </Button>
            <Button asChild variant="primary" size="md">
              <Link to="/datasets/new" search={{ example: undefined }}>
                <Plus />
                Train on your data
              </Link>
            </Button>
            <Button asChild variant="secondary" size="md">
              <Link to="/datasets/new" search={{ example: "1" }}>
                <Zap />
                Try a working example
              </Link>
            </Button>
          </>
        }
        statusBar={
          <>
            <ReadoutItem label="UPDATED" value={lastRefreshed} />
            <ReadoutSep />
            <ReadoutItem label="BACKEND" value={backend.data?.name ?? "—"} />
            <ReadoutSep />
            <ReadoutItem label="DTYPE" value={backend.data?.capabilities.preferred_dtype_str ?? "—"} />
            <ReadoutSep />
            <ReadoutItem label="RUNS" value={String(items.length)} />
          </>
        }
      />

      <div className="px-5 py-5">
        {readiness.data?.status === "blocked" ? <div className="mb-4 flex flex-wrap items-center justify-between gap-3 border border-warning/35 bg-warning/5 px-4 py-3"><div className="flex items-start gap-2"><AlertTriangle className="mt-0.5 h-4 w-4 text-warning" /><div><p className="text-sm font-medium text-fg">{readiness.data.display_status}</p><p className="mt-0.5 text-xs text-fg-muted">{readiness.data.summary}</p></div></div><Button asChild variant="primary" size="sm"><Link to="/setup">Fix setup</Link></Button></div> : null}
        <StatRibbon items={items} />

        <div className="mt-4 grid min-h-[520px] border-y border-border-subtle xl:grid-cols-[260px_minmax(0,1fr)_300px]">
          <CurrentWork items={activity.data?.items ?? []} loading={activity.isLoading} />
          <div className="min-w-0 border-b border-border-subtle xl:border-b-0 xl:border-r">
            <RecentRunsCard items={items} loading={runs.isLoading} />
          </div>
          <aside className="bg-bg-subtle/20">
            <AttentionList items={activity.data?.items ?? []} />
            <RecentDecisions items={decisions.data?.items ?? []} />
            <SystemCard />
          </aside>
        </div>
      </div>
    </>
  );
}

/* ------------------------------------------------------------------------
 * Stat ribbon — four mono tiles, the operator's "vital signs" digest.
 * ---------------------------------------------------------------------- */

function StatRibbon({ items }: { items: RunListItem[] }) {
  const total = items.length;
  const passed = items.filter((r) => r.effectiveness?.verdict === "passed").length;
  const updated = items.filter((r) => Boolean(r.weights_updated)).length;
  const lastLoss = items
    .map((r) => (typeof r.final_train_loss === "number" ? r.final_train_loss : null))
    .find((v): v is number => v != null);

  const tiles: StatTileSpec[] = [
    { label: "RECENT RUNS", value: compactNumber(total, 0) },
    {
      label: "EFFECTIVENESS",
      value: total ? `${passed}/${total}` : "—",
      hint: total ? `${Math.round((passed / total) * 100)}%` : undefined,
      tone: passed === total && total > 0 ? "success" : passed > 0 ? "neutral" : "neutral",
    },
    {
      label: "WEIGHTS UPDATED",
      value: total ? `${updated}/${total}` : "—",
      hint: total ? `${Math.round((updated / total) * 100)}%` : undefined,
    },
    {
      label: "LAST LOSS",
      value: lastLoss != null ? lastLoss.toFixed(3) : "—",
    },
  ];

  return (
    <div className="grid grid-cols-2 divide-x divide-y divide-border-subtle border border-border-subtle lg:grid-cols-4 lg:divide-y-0">
      {tiles.map((tile) => (
        <StatTile key={tile.label} {...tile} />
      ))}
    </div>
  );
}

type StatTileSpec = {
  label: string;
  value: string;
  hint?: string;
  tone?: "success" | "warning" | "danger" | "neutral";
};

function StatTile({ label, value, hint, tone }: StatTileSpec) {
  return (
    <div className="bg-surface/25 px-3.5 py-3">
        <div className="text-[10px] font-medium uppercase tracking-[0.14em] text-fg-disabled">
          {label}
        </div>
        <div className="mt-1.5 flex items-baseline gap-2">
          <span
            className={cn(
              "font-mono text-[22px] leading-none tabular-nums tracking-tight",
              tone === "success" ? "text-success" : "text-fg",
            )}
          >
            {value}
          </span>
          {hint ? (
            <span className="font-mono text-[11px] text-fg-subtle">{hint}</span>
          ) : null}
        </div>
    </div>
  );
}

/* ------------------------------------------------------------------------
 * Recent runs — dense, sortable-feeling table with mono identifiers.
 * ---------------------------------------------------------------------- */

function RecentRunsCard({ items, loading }: { items: RunListItem[]; loading: boolean }) {
  return (
    <section>
      <header className="flex items-center justify-between border-b border-border-subtle px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="text-[9.5px] font-medium uppercase tracking-[0.13em] text-fg-disabled">Completed and active</span>
          <span className="text-[13px] font-medium text-fg">Recent runs</span>
        </div>
        <Button asChild variant="ghost" size="sm">
          <Link to="/runs">
            View all
            <ArrowUpRight />
          </Link>
        </Button>
      </header>
      <div>
        {loading ? (
          <div className="space-y-px">
            {[0, 1, 2, 3].map((i) => (
              <div key={i} className="h-10 animate-pulse bg-surface-hover/30" />
            ))}
          </div>
        ) : items.length === 0 ? (
          <EmptyRunsState />
        ) : (
          <table className="w-full text-[13px]">
            <thead>
              <tr className="border-b border-border-subtle">
                <Th>Run</Th>
                <Th>Modality</Th>
                <Th>Status</Th>
                <Th align="right">Cycles</Th>
                <Th align="right">Loss</Th>
                <Th align="right">When</Th>
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
                const loss =
                  typeof run.final_train_loss === "number"
                    ? run.final_train_loss.toFixed(3)
                    : "—";
                return (
                  <tr
                    key={run.run_id}
                    className="group border-b border-border-subtle last:border-0 hover:bg-surface-hover/40 transition-colors"
                  >
                    <Td>
                      <Link
                        to="/runs/$runId"
                        params={{ runId: run.run_id }}
                        className="font-mono text-[12px] text-accent group-hover:underline"
                      >
                        {run.run_id.slice(0, 18) || "—"}
                      </Link>
                      <div className="text-[11px] text-fg-disabled truncate max-w-[28ch] mt-0.5">
                        {run.model_name}
                      </div>
                    </Td>
                    <Td className="text-fg-muted capitalize">{run.modality}</Td>
                    <Td>
                      <Badge tone={tone} dot size="sm">
                        {verdict ?? "pending"}
                      </Badge>
                    </Td>
                    <Td align="right" mono>
                      {run.cycles_executed ?? "—"}
                    </Td>
                    <Td align="right" mono className="text-fg">
                      {loss}
                    </Td>
                    <Td align="right" className="text-fg-muted whitespace-nowrap">
                      {relativeTime(run.created_at)}
                    </Td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </section>
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

function EmptyRunsState() {
  return (
    <div className="flex flex-col items-center justify-center px-6 py-12 text-center">
      <div className="flex h-9 w-9 items-center justify-center rounded-md border border-border-subtle bg-surface">
        <Zap className="h-4 w-4 text-fg-subtle" />
      </div>
      <div className="mt-3 text-[13px] font-medium text-fg">No runs yet</div>
      <div className="mt-1 text-xs text-fg-muted max-w-[36ch]">Start with your own files or use a small verified example to prove the complete path.</div>
      <Button asChild variant="primary" size="sm" className="mt-3.5">
        <Link to="/datasets/new" search={{ example: undefined }}>Train on your data</Link>
      </Button>
    </div>
  );
}

function CurrentWork({ items, loading }: { items: ActivityItem[]; loading: boolean }) {
  const active = items.filter((item) => ["queued", "running", "preparing", "blocked", "awaiting_review"].includes(item.status)).slice(0, 10);
  return (
    <aside className="border-b border-border-subtle bg-bg-subtle/20 xl:border-b-0 xl:border-r">
      <div className="border-b border-border-subtle px-4 py-3"><div className="text-[9.5px] font-medium uppercase tracking-[0.13em] text-fg-disabled">Current work</div><p className="mt-1 text-[10.5px] text-fg-subtle">Queue position and attention required.</p></div>
      <div className="divide-y divide-border-subtle">
        {active.map((item) => <CurrentWorkRow key={item.id} item={item} />)}
        {loading ? <OverviewMessage icon={Activity} label="Loading workstation" /> : null}
        {!loading && !active.length ? <OverviewMessage icon={CheckCircle2} label="Workstation is clear" detail="New training and evaluation work will appear here." success /> : null}
      </div>
      <div className="border-t border-border-subtle px-4 py-3"><Button asChild size="sm" variant="ghost"><Link to="/sweeps"><Plus /> New experiment</Link></Button></div>
    </aside>
  );
}

function CurrentWorkRow({ item }: { item: ActivityItem }) {
  const progress = typeof item.progress_percent === "number" ? item.progress_percent : item.progress_total ? (item.progress_current ?? 0) / item.progress_total * 100 : null;
  const Icon = item.status === "awaiting_review" ? AlertTriangle : item.status === "blocked" ? Clock3 : item.status === "running" ? Activity : CircleDashed;
  return <div className={cn("px-4 py-3", item.status === "awaiting_review" && "bg-warning-bg/30")}><div className="flex items-start gap-2.5"><Icon className={cn("mt-0.5 h-3.5 w-3.5 shrink-0", item.status === "awaiting_review" || item.status === "blocked" ? "text-warning" : item.status === "running" ? "text-accent" : "text-fg-disabled")} /><div className="min-w-0 flex-1"><div className="truncate text-[11.5px] font-medium capitalize text-fg">{item.title || item.kind.replaceAll("_", " ")}</div><div className="mt-1 flex items-center justify-between gap-2 font-mono text-[9px] uppercase text-fg-disabled"><span className="truncate">{item.status.replaceAll("_", " ")}</span><span>{item.queue_position ? `queue ${item.queue_position}` : item.stage || ""}</span></div>{progress != null ? <div className="mt-2 h-0.5 bg-surface-pressed"><div className="h-full bg-accent transition-[width] motion-reduce:transition-none" style={{ width: `${Math.min(100, Math.max(0, progress))}%` }} /></div> : null}</div></div></div>;
}

function AttentionList({ items }: { items: ActivityItem[] }) {
  const attention = items.filter((item) => ["awaiting_review", "failed", "interrupted", "needs_reconciliation", "blocked"].includes(item.status)).slice(0, 5);
  return <section><div className="flex items-center justify-between border-b border-border-subtle px-4 py-3"><div><div className="text-[9.5px] font-medium uppercase tracking-[0.13em] text-fg-disabled">Attention required</div><div className="mt-1 text-[10.5px] text-fg-subtle">Review gates and recover failures.</div></div><Badge tone={attention.length ? "warning" : "neutral"} size="sm">{attention.length}</Badge></div><div className="divide-y divide-border-subtle">{attention.map((item) => <div key={item.id} className="px-4 py-2.5"><div className="flex items-center gap-2"><AlertTriangle className="h-3 w-3 shrink-0 text-warning" /><span className="truncate text-[10.5px] font-medium text-fg">{item.title || item.kind.replaceAll("_", " ")}</span></div><div className="mt-1 pl-5 text-[9.5px] capitalize text-fg-disabled">{item.status.replaceAll("_", " ")}</div></div>)}{!attention.length ? <OverviewMessage icon={CheckCircle2} label="Nothing needs review" success /> : null}</div></section>;
}

function RecentDecisions({ items }: { items: ResearchDecisionRecord[] }) {
  return <section className="border-t border-border-subtle"><div className="border-b border-border-subtle px-4 py-3"><div className="text-[9.5px] font-medium uppercase tracking-[0.13em] text-fg-disabled">Recent decisions</div><p className="mt-1 text-[10.5px] text-fg-subtle">Append-only evidence selections.</p></div><div className="divide-y divide-border-subtle">{items.slice(0, 4).map((item) => <div key={item.id} className="px-4 py-2.5"><div className="line-clamp-2 text-[10.5px] leading-relaxed text-fg-muted">{item.rationale}</div><div className="mt-1 font-mono text-[9px] text-fg-disabled">{relativeTime(item.created_at)} · {String(item.selected_subject.trial_id ?? item.selected_subject.run_id ?? item.id).slice(0, 12)}</div></div>)}{!items.length ? <OverviewMessage icon={CircleDashed} label="No research decisions yet" detail="Analyze a completed cohort to record one." /> : null}</div></section>;
}

function OverviewMessage({ icon: Icon, label, detail, success }: { icon: typeof Activity; label: string; detail?: string; success?: boolean }) {
  return <div className="px-4 py-6 text-center"><Icon className={cn("mx-auto h-4 w-4 text-fg-disabled", success && "text-success")} /><div className="mt-2 text-[10.5px] text-fg-muted">{label}</div>{detail ? <div className="mt-1 text-[9.5px] leading-relaxed text-fg-disabled">{detail}</div> : null}</div>;
}

/* ------------------------------------------------------------------------
 * System card — backend identity + capabilities.
 * ---------------------------------------------------------------------- */

function SystemCard() {
  const { data, isLoading } = useBackendInfo();

  return (
    <section className="border-t border-border-subtle">
      <header className="flex items-center justify-between border-b border-border-subtle px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="text-[9.5px] font-medium uppercase tracking-[0.13em] text-fg-disabled">System</span>
          <span className="text-[12.5px] font-medium text-fg">Compute</span>
        </div>
        <Cpu className="h-3.5 w-3.5 text-fg-disabled" />
      </header>
      <div className="divide-y divide-border-subtle text-[13px]">
        {isLoading ? (
          <div className="space-y-2 p-3.5">
            <div className="h-3 animate-pulse rounded-sm bg-surface-hover" />
            <div className="h-3 w-3/4 animate-pulse rounded-sm bg-surface-hover" />
            <div className="h-3 w-1/2 animate-pulse rounded-sm bg-surface-hover" />
          </div>
        ) : !data ? (
          <div className="px-3.5 py-3.5 text-fg-muted">Backend unreachable.</div>
        ) : (
          <>
            <SysRow label="Backend" value={data.name} mono />
            <SysRow label="Device" value={data.device} mono />
            {data.chip ? (
              <SysRow
                label="Apple chip"
                value={
                  data.chip.gpu_cores != null
                    ? `${data.chip.brand}, ${data.chip.gpu_cores} GPU cores`
                    : data.chip.brand
                }
              />
            ) : null}
            <SysRow label="Default dtype" value={data.capabilities.preferred_dtype_str} mono />
            <SysRow label="Attention" value={data.capabilities.preferred_attn_impl} mono />
            <SysRow
              label="MLX runtime"
              tone={data.mlx_readiness?.executable ? "success" : "neutral"}
              value={data.mlx_readiness?.executable ? "ready" : data.mlx_readiness?.status ?? "unavailable"}
            />
            <SysRow
              label="Apple Neural Accelerators (experimental)"
              tone={data.capabilities.supports_neural_accelerators ? "success" : "neutral"}
              value={data.capabilities.supports_neural_accelerators ? "available" : "not used by MPS/MLX"}
            />
            <SysRow
              label="4-bit quant"
              tone={data.capabilities.supports_4bit ? "success" : "neutral"}
              value={data.capabilities.supports_4bit ? "available" : "unavailable"}
            />
            <SysRow
              label="Training"
              tone={data.capabilities.supports_training ? "success" : "warning"}
              value={data.capabilities.supports_training ? "supported" : "unsupported"}
            />
          </>
        )}
      </div>
    </section>
  );
}

function SysRow({
  label,
  value,
  mono,
  tone,
}: {
  label: string;
  value: string;
  mono?: boolean;
  tone?: "success" | "warning" | "danger" | "neutral";
}) {
  return (
    <div className="flex items-center justify-between gap-2 px-3.5 py-2">
      <span className="text-fg-muted text-[12px]">{label}</span>
      {tone && tone !== "neutral" ? (
        <Badge tone={tone} dot size="sm">
          {value}
        </Badge>
      ) : (
        <span className={cn(mono ? "font-mono text-fg" : "text-fg", "text-[12px]")}>
          {value}
        </span>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------------
 * Topbar status row primitives — mono readouts separated by middle dots.
 * ---------------------------------------------------------------------- */

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
