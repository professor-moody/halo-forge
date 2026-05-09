import { createFileRoute, Link } from "@tanstack/react-router";
import {
  ArrowUpRight,
  Cpu,
  Plus,
  RefreshCw,
  Zap,
} from "lucide-react";
import { useDashboard, useRuns, useBackendInfo } from "@/lib/hooks";
import { Topbar } from "@/components/shell";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle, CardEyebrow } from "@/components/ui/card";
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
              <Link to="/train">
                <Plus />
                New run
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

      <div className="px-5 py-5 space-y-4">
        <StatRibbon items={items} />

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
          <div className="lg:col-span-2">
            <RecentRunsCard items={items} loading={runs.isLoading} />
          </div>
          <div>
            <SystemCard />
          </div>
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
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-2.5">
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
    <Card className="bg-surface/80">
      <CardContent className="px-3.5 py-3">
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
      </CardContent>
    </Card>
  );
}

/* ------------------------------------------------------------------------
 * Recent runs — dense, sortable-feeling table with mono identifiers.
 * ---------------------------------------------------------------------- */

function RecentRunsCard({ items, loading }: { items: RunListItem[]; loading: boolean }) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardEyebrow>ACTIVITY</CardEyebrow>
          <CardTitle>Recent runs</CardTitle>
        </div>
        <Button asChild variant="ghost" size="sm">
          <Link to="/runs">
            View all
            <ArrowUpRight />
          </Link>
        </Button>
      </CardHeader>
      <CardContent className="p-0">
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

function EmptyRunsState() {
  return (
    <div className="flex flex-col items-center justify-center px-6 py-12 text-center">
      <div className="flex h-9 w-9 items-center justify-center rounded-md border border-border-subtle bg-surface">
        <Zap className="h-4 w-4 text-fg-subtle" />
      </div>
      <div className="mt-3 text-[13px] font-medium text-fg">No runs yet</div>
      <div className="mt-1 text-xs text-fg-muted max-w-[36ch]">
        Launch a training job to populate this list. RAFT and SFT both surface here.
      </div>
      <Button asChild variant="primary" size="sm" className="mt-3.5">
        <Link to="/train">Start a run</Link>
      </Button>
    </div>
  );
}

/* ------------------------------------------------------------------------
 * System card — backend identity + capabilities.
 * ---------------------------------------------------------------------- */

function SystemCard() {
  const { data, isLoading } = useBackendInfo();

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardEyebrow>SYSTEM</CardEyebrow>
          <CardTitle>Compute</CardTitle>
        </div>
        <Cpu className="h-3.5 w-3.5 text-fg-disabled" />
      </CardHeader>
      <CardContent className="text-[13px] divide-y divide-border-subtle p-0">
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
              label="Neural Accelerators"
              tone={data.capabilities.supports_neural_accelerators ? "success" : "neutral"}
              value={data.capabilities.supports_neural_accelerators ? "available" : "unavailable"}
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
      </CardContent>
    </Card>
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
