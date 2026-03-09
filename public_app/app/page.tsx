import Link from "next/link";

import {
  ActionLink,
  AppShell,
  EmptyState,
  MetricTile,
  SectionCard,
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
      subtitle="See live work, qualification truth, and the runs that need action without digging through diagnostics."
      statusItems={[
        {
          label: "Qualification status",
          value: dashboard?.readiness_tier ?? "unavailable",
          tone:
            dashboard?.readiness_tier === "production_ready"
              ? "success"
              : dashboard?.readiness_tier === "qualified"
                ? "warning"
                : "neutral",
        },
        {
          label: "Active runs",
          value: String(dashboard?.active_runs_count ?? 0),
          tone: "neutral",
        },
        {
          label: "Needs attention",
          value: String(dashboard?.attention_count ?? 0),
          tone: (dashboard?.attention_count ?? 0) > 0 ? "warning" : "success",
        },
      ]}
      headerActions={
        <div className="action-strip">
          <ActionLink href="/train" label="Start training" tone="primary" />
          <ActionLink href="/results" label="Review outcomes" tone="secondary" />
        </div>
      }
    >
      <SectionCard
        title="System summary"
        subtitle="The product should tell you where the work is, what the system trusts, and what to do next."
        eyebrow="Control center"
        className="surface-hero"
      >
        <div className="summary-strip">
          <MetricTile
            label="Readiness"
            value={dashboard?.readiness_tier ?? "unavailable"}
            meta={dashboard?.generated_at ? `Updated ${dashboard.generated_at}` : "Qualification data unavailable"}
          />
          <MetricTile
            label="Production ready"
            value={`${dashboard?.production_ready_count ?? 0}/${dashboard?.modality_count ?? 0}`}
            meta="Modalities passing deterministic launch, update, artifact, resume, and eval checks"
          />
          <MetricTile
            label="Active runs"
            value={String(dashboard?.active_runs_count ?? 0)}
            meta="Jobs that are still consuming compute or attention"
          />
          <MetricTile
            label="Needs attention"
            value={String(dashboard?.attention_count ?? 0)}
            meta="Low-yield, failed, or recovery-ready runs"
          />
        </div>
      </SectionCard>

      <div className="dashboard-grid">
        <SectionCard title="Active runs" subtitle="Running or pending jobs, with the minimum context needed to decide whether to keep watching or intervene.">
          <div className="data-table">
            {dashboard?.active_runs.length ? (
              dashboard.active_runs.map((run) => (
                <Link key={run.id} href={`/runs/${encodeURIComponent(run.id)}`} className="data-row">
                  <div className="row-main">
                    <div className="row-title">
                      <h3>{run.modality.toUpperCase()} · {run.model_name}</h3>
                      <p>{run.headline}</p>
                    </div>
                    <StatusChip
                      tone={run.status === "running" ? "success" : "warning"}
                      label={run.status}
                    />
                  </div>
                  <div className="row-metrics">
                    <div className="row-detail">
                      <div className="cell-label">Progress</div>
                      <strong>{run.metrics_summary.progress_percent.toFixed(1)}%</strong>
                    </div>
                    <div className="row-detail">
                      <div className="cell-label">Update steps</div>
                      <strong>{run.metrics_summary.update_steps}</strong>
                    </div>
                    <div className="row-detail">
                      <div className="cell-label">Keep rate</div>
                      <strong>{percent(run.metrics_summary.keep_rate)}</strong>
                    </div>
                    <div className="row-detail">
                      <div className="cell-label">Recommended next step</div>
                      <strong>{run.next_step}</strong>
                    </div>
                  </div>
                </Link>
              ))
            ) : (
              <EmptyState
                title="No active runs"
                body="Start a run from Training or review completed outcomes below."
              />
            )}
          </div>
        </SectionCard>

        <SectionCard title="Needs attention" subtitle="The small set of runs that currently deserve a closer look or a recovery action.">
          <div className="attention-grid">
            {dashboard?.attention_items.length ? (
              dashboard.attention_items.map((item) => (
                <Link key={item.id} href={`/runs/${encodeURIComponent(item.id)}`} className="attention-card">
                  <div className="row-main">
                    <div className="row-title">
                      <h3>{item.modality.toUpperCase()} · {item.headline}</h3>
                      <p>{item.why_it_matters}</p>
                    </div>
                    <StatusChip tone={item.confidence_tone} label={item.next_step} />
                  </div>
                </Link>
              ))
            ) : (
              <EmptyState
                title="No urgent intervention"
                body="The current run set is not surfacing high-priority recovery issues."
              />
            )}
          </div>
        </SectionCard>
      </div>

      <SectionCard
        title="Recent outcomes"
        subtitle="Completed runs distilled to outcome, cause, and next action."
      >
        <div className="data-table">
          {dashboard?.recent_outcomes.length ? (
            dashboard.recent_outcomes.map((run) => (
              <Link key={run.id} href={`/runs/${encodeURIComponent(run.id)}`} className="data-row">
                <div className="row-main">
                  <div className="row-title">
                    <h3>{run.modality.toUpperCase()} · {run.model_name}</h3>
                    <p>{run.user_summary.why_it_matters}</p>
                  </div>
                  <StatusChip tone={run.user_summary.confidence_tone} label={run.headline} />
                </div>
                <div className="row-metrics">
                  <div className="row-detail">
                    <div className="cell-label">Top issue</div>
                    <strong>{run.top_issue ?? "—"}</strong>
                  </div>
                  <div className="row-detail">
                    <div className="cell-label">Next step</div>
                    <strong>{run.next_step}</strong>
                  </div>
                  <div className="row-detail">
                    <div className="cell-label">Update steps</div>
                    <strong>{run.metrics_summary.update_steps}</strong>
                  </div>
                  <div className="row-detail">
                    <div className="cell-label">Final loss</div>
                    <strong>{run.metrics_summary.final_train_loss ?? "—"}</strong>
                  </div>
                </div>
              </Link>
            ))
          ) : (
            <EmptyState
              title="No completed outcomes yet"
              body="Once training summaries land, this becomes the fastest way to review what happened."
            />
          )}
        </div>
      </SectionCard>
    </AppShell>
  );
}
