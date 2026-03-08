import Link from "next/link";

import {
  ActionLink,
  AppShell,
  EmptyState,
  SectionCard,
  StatTile,
  StatusChip,
} from "../components/ui";
import { apiGet } from "../lib/api";

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
    primary_action?: { label: string };
    metrics_summary: {
      progress_percent: number;
      keep_rate?: number | null;
      update_steps: number;
      final_train_loss?: number | null;
    };
  }>;
  attention_items: Array<{
    id: string;
    modality: string;
    headline: string;
    why_it_matters: string;
    next_step: string;
    confidence_tone: string;
    primary_action?: { label: string };
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
      progress_percent: number;
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
  if (typeof value !== "number") {
    return "—";
  }
  return `${Math.round(value * 100)}%`;
}

export default async function HomePage() {
  const dashboard = await getDashboard();

  return (
    <AppShell
      title="Overview"
      subtitle="Active work, readiness, recent outcomes, and the next action that matters."
      statusItems={[
        { label: "Readiness", value: dashboard?.readiness_tier ?? "unavailable", tone: dashboard?.readiness_tier === "production_ready" ? "success" : dashboard?.readiness_tier === "qualified" ? "warning" : "neutral" },
        { label: "Active runs", value: String(dashboard?.active_runs_count ?? 0), tone: "neutral" },
        { label: "Needs attention", value: String(dashboard?.attention_count ?? 0), tone: (dashboard?.attention_count ?? 0) > 0 ? "warning" : "success" },
      ]}
    >
      <SectionCard
        title="System summary"
        subtitle="What is running, what is qualified, and what needs intervention right now."
        actions={<ActionLink href="/train" label="Start training" tone="primary" />}
      >
        <div className="stat-grid">
          <StatTile label="Active runs" value={String(dashboard?.active_runs_count ?? 0)} hint="Pending or running jobs" />
          <StatTile label="Needs attention" value={String(dashboard?.attention_count ?? 0)} hint="Low-yield, failed, or recovery-ready runs" />
          <StatTile label="Production ready" value={`${dashboard?.production_ready_count ?? 0}/${dashboard?.modality_count ?? 0}`} hint="Qualified modalities" />
          <StatTile label="Qualification" value={dashboard?.readiness_tier ?? "unavailable"} hint={dashboard?.generated_at ? `Updated ${dashboard.generated_at}` : "Readiness data unavailable"} />
        </div>
      </SectionCard>

      <div className="grid-dashboard">
        <SectionCard title="Active runs" subtitle="The jobs that are currently consuming attention or hardware.">
          <div className="run-list">
            {dashboard?.active_runs.length ? (
              dashboard.active_runs.map((run) => (
                <Link key={run.id} href={`/runs/${encodeURIComponent(run.id)}`} className="table-row">
                  <div className="table-row-header">
                    <div className="table-row-title">
                      <h3>{run.modality.toUpperCase()} · {run.model_name}</h3>
                      <p>{run.headline}</p>
                    </div>
                    <StatusChip tone={run.status === "running" ? "success" : "warning"} label={run.status} />
                  </div>
                  <div className="table-row-metrics">
                    <div>
                      <div className="cell-label">Progress</div>
                      <div className="cell-value">{run.metrics_summary.progress_percent.toFixed(1)}%</div>
                    </div>
                    <div>
                      <div className="cell-label">Update steps</div>
                      <div className="cell-value">{run.metrics_summary.update_steps}</div>
                    </div>
                    <div>
                      <div className="cell-label">Keep rate</div>
                      <div className="cell-value">{percent(run.metrics_summary.keep_rate)}</div>
                    </div>
                    <div>
                      <div className="cell-label">Next step</div>
                      <div className="cell-value">{run.next_step}</div>
                    </div>
                  </div>
                </Link>
              ))
            ) : (
              <EmptyState
                title="No active runs"
                body="Launch a new training job or review the most recent outcomes below."
              />
            )}
          </div>
        </SectionCard>

        <SectionCard title="Needs attention" subtitle="The highest-signal issues worth looking at next.">
          <div className="attention-list">
            {dashboard?.attention_items.length ? (
              dashboard.attention_items.map((item) => (
                <Link key={item.id} href={`/runs/${encodeURIComponent(item.id)}`} className="attention-item">
                  <div className="table-row-header">
                    <h3>{item.modality.toUpperCase()} · {item.headline}</h3>
                    <StatusChip tone={item.confidence_tone} label={item.next_step} />
                  </div>
                  <p>{item.why_it_matters}</p>
                </Link>
              ))
            ) : (
              <EmptyState
                title="No urgent remediation"
                body="Recent runs are not surfacing obvious recovery issues."
              />
            )}
          </div>
        </SectionCard>
      </div>

      <SectionCard title="Recent outcomes" subtitle="Completed training runs, condensed to outcome, cause, and next step.">
        <div className="run-list">
          {dashboard?.recent_outcomes.length ? (
            dashboard.recent_outcomes.map((run) => (
              <Link key={run.id} href={`/runs/${encodeURIComponent(run.id)}`} className="table-row">
                <div className="table-row-header">
                  <div className="table-row-title">
                    <h3>{run.modality.toUpperCase()} · {run.model_name}</h3>
                    <p>{run.user_summary.why_it_matters}</p>
                  </div>
                  <StatusChip tone={run.user_summary.confidence_tone} label={run.headline} />
                </div>
                <div className="table-row-metrics">
                  <div>
                    <div className="cell-label">Next step</div>
                    <div className="cell-value">{run.next_step}</div>
                  </div>
                  <div>
                    <div className="cell-label">Top issue</div>
                    <div className="cell-value">{run.top_issue ?? "—"}</div>
                  </div>
                  <div>
                    <div className="cell-label">Update steps</div>
                    <div className="cell-value">{run.metrics_summary.update_steps}</div>
                  </div>
                  <div>
                    <div className="cell-label">Final loss</div>
                    <div className="cell-value">{run.metrics_summary.final_train_loss ?? "—"}</div>
                  </div>
                </div>
              </Link>
            ))
          ) : (
            <EmptyState
              title="No completed outcomes yet"
              body="As training summaries land, this table will become the fastest way to review what happened."
            />
          )}
        </div>
      </SectionCard>
    </AppShell>
  );
}
