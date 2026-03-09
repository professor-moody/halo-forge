import Link from "next/link";

import {
  ActionLink,
  AppShell,
  EmptyState,
  SectionCard,
  StatusBadge,
} from "@/components/app-ui";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { apiGet } from "@/lib/api";

type DashboardResponse = {
  readiness_tier: string;
  generated_at?: string | null;
  active_runs_count: number;
  attention_count: number;
  production_ready_count: number;
  modality_count: number;
  active_runs: Array<{
    id: string;
    modality: string;
    model_name: string;
    status: string;
    headline: string;
    next_step: string;
    metrics_summary: {
      progress_percent: number;
      keep_rate?: number | null;
      update_steps: number;
    };
  }>;
  attention_items: Array<{
    id: string;
    modality: string;
    headline: string;
    why_it_matters: string;
    next_step: string;
    confidence_tone: string;
  }>;
  recent_outcomes: Array<{
    id: string;
    modality: string;
    model_name: string;
    headline: string;
    next_step: string;
    top_issue?: string | null;
    user_summary: { confidence_tone: string; why_it_matters: string };
    metrics_summary: {
      keep_rate?: number | null;
      update_steps: number;
      final_train_loss?: number | null;
    };
  }>;
};

async function getDashboard() {
  try {
    return await apiGet<DashboardResponse>("/dashboard");
  } catch {
    return null;
  }
}

function percent(value?: number | null) {
  if (typeof value !== "number") return "—";
  return `${Math.round(value * 100)}%`;
}

export default async function HomePage() {
  const dashboard = await getDashboard();

  return (
    <AppShell
      title="Overview"
      subtitle="Live work, qualification status, and runs that need action."
      statusItems={[
        {
          label: "Qualification",
          value: dashboard?.readiness_tier ?? "unavailable",
          tone:
            dashboard?.readiness_tier === "production_ready"
              ? "success"
              : dashboard?.readiness_tier === "qualified"
                ? "warning"
                : "neutral",
        },
        {
          label: "Active",
          value: String(dashboard?.active_runs_count ?? 0),
          tone: "neutral",
        },
        {
          label: "Attention",
          value: String(dashboard?.attention_count ?? 0),
          tone: (dashboard?.attention_count ?? 0) > 0 ? "warning" : "success",
        },
      ]}
      headerActions={
        <div className="flex gap-2">
          <ActionLink href="/train" label="Start training" tone="primary" />
          <ActionLink href="/results" label="Review outcomes" tone="secondary" />
        </div>
      }
    >
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">System summary</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-4 gap-4">
            <Stat label="Readiness" value={dashboard?.readiness_tier ?? "unavailable"} sub={dashboard?.generated_at ? `Updated ${dashboard.generated_at}` : undefined} />
            <Stat label="Production ready" value={`${dashboard?.production_ready_count ?? 0}/${dashboard?.modality_count ?? 0}`} sub="Modalities passing all checks" />
            <Stat label="Active runs" value={String(dashboard?.active_runs_count ?? 0)} sub="Consuming compute" />
            <Stat label="Needs attention" value={String(dashboard?.attention_count ?? 0)} sub="Low-yield or failed" />
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-[1.4fr_1fr] gap-4">
        <SectionCard title="Active runs" subtitle="Running or pending jobs.">
          {dashboard?.active_runs.length ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Run</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="text-right">Progress</TableHead>
                  <TableHead className="text-right">Steps</TableHead>
                  <TableHead className="text-right">Keep rate</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {dashboard.active_runs.map((run) => (
                  <TableRow key={run.id}>
                    <TableCell>
                      <Link href={`/runs/${encodeURIComponent(run.id)}`} className="hover:underline">
                        <span className="font-medium">{run.modality.toUpperCase()}</span>
                        <span className="text-muted-foreground"> · {run.model_name}</span>
                      </Link>
                    </TableCell>
                    <TableCell>
                      <Badge variant={run.status === "running" ? "success" : "warning"}>{run.status}</Badge>
                    </TableCell>
                    <TableCell className="text-right tabular-nums">{run.metrics_summary.progress_percent.toFixed(1)}%</TableCell>
                    <TableCell className="text-right tabular-nums">{run.metrics_summary.update_steps}</TableCell>
                    <TableCell className="text-right tabular-nums">{percent(run.metrics_summary.keep_rate)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <EmptyState title="No active runs" body="Start a run from Training." />
          )}
        </SectionCard>

        <SectionCard title="Needs attention" subtitle="Runs requiring a closer look.">
          {dashboard?.attention_items.length ? (
            <div className="space-y-1.5">
              {dashboard.attention_items.map((item) => (
                <Link
                  key={item.id}
                  href={`/runs/${encodeURIComponent(item.id)}`}
                  className="block rounded-md border border-border border-l-2 border-l-amber-500 px-3 py-2 hover:bg-accent/50 transition-colors"
                >
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <div className="text-sm font-medium">{item.modality.toUpperCase()} · {item.headline}</div>
                      <p className="text-xs text-muted-foreground mt-0.5">{item.why_it_matters}</p>
                    </div>
                    <Badge variant="warning" className="shrink-0">{item.next_step}</Badge>
                  </div>
                </Link>
              ))}
            </div>
          ) : (
            <EmptyState title="No urgent items" body="No high-priority recovery issues." />
          )}
        </SectionCard>
      </div>

      <SectionCard title="Recent outcomes" subtitle="Completed runs distilled to outcome and next action.">
        {dashboard?.recent_outcomes.length ? (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Run</TableHead>
                <TableHead>Outcome</TableHead>
                <TableHead>Top issue</TableHead>
                <TableHead>Next step</TableHead>
                <TableHead className="text-right">Steps</TableHead>
                <TableHead className="text-right">Final loss</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {dashboard.recent_outcomes.map((run) => (
                <TableRow key={run.id}>
                  <TableCell>
                    <Link href={`/runs/${encodeURIComponent(run.id)}`} className="hover:underline">
                      <span className="font-medium">{run.modality.toUpperCase()}</span>
                      <span className="text-muted-foreground"> · {run.model_name}</span>
                    </Link>
                  </TableCell>
                  <TableCell>
                    <StatusBadge tone={run.user_summary.confidence_tone} label={run.headline} />
                  </TableCell>
                  <TableCell className="text-muted-foreground">{run.top_issue ?? "—"}</TableCell>
                  <TableCell className="text-muted-foreground">{run.next_step}</TableCell>
                  <TableCell className="text-right tabular-nums">{run.metrics_summary.update_steps}</TableCell>
                  <TableCell className="text-right tabular-nums">{run.metrics_summary.final_train_loss ?? "—"}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        ) : (
          <EmptyState title="No completed outcomes" body="Outcomes appear here after training summaries are written." />
        )}
      </SectionCard>
    </AppShell>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="text-lg font-semibold text-foreground mt-0.5">{value}</div>
      {sub ? <div className="text-xs text-muted-foreground mt-0.5">{sub}</div> : null}
    </div>
  );
}
