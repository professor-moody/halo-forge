import Link from "next/link";

import { AppShell, EmptyState, SectionCard, StatusChip } from "../../components/ui";
import { apiGet } from "../../lib/api";

type ResultsResponse = {
  items: Array<{
    id: string;
    modality: string;
    model_name: string;
    headline: string;
    next_step: string;
    top_issue?: string | null;
    user_summary: {
      why_it_matters: string;
      confidence_tone: string;
    };
    metrics_summary: {
      keep_rate?: number | null;
      update_steps: number;
      final_train_loss?: number | null;
      eval_metric_name?: string;
      eval_metric_value?: number | null;
    };
  }>;
};

async function getResults() {
  try {
    return await apiGet<ResultsResponse>("/results/training?include_research=false");
  } catch {
    return { items: [] };
  }
}

export default async function ResultsPage() {
  const payload = await getResults();

  return (
    <AppShell
      title="Results"
      subtitle="Dense review of completed runs, causes, and the next action each one suggests."
      statusItems={[
        { label: "Completed runs", value: String(payload.items.length), tone: "neutral" },
        { label: "Focus", value: "outcome review", tone: "success" },
      ]}
    >
      <SectionCard title="Training outcomes" subtitle="Each row answers whether the run worked, what went wrong, and what to do next.">
        <div className="run-list">
          {payload.items.length ? (
            payload.items.map((item) => (
              <Link key={item.id} href={`/runs/${encodeURIComponent(item.id)}`} className="table-row">
                <div className="table-row-header">
                  <div className="table-row-title">
                    <h3>{item.modality.toUpperCase()} · {item.model_name}</h3>
                    <p>{item.user_summary.why_it_matters}</p>
                  </div>
                  <StatusChip tone={item.user_summary.confidence_tone} label={item.headline} />
                </div>
                <div className="table-row-metrics">
                  <div>
                    <div className="cell-label">Next step</div>
                    <div className="cell-value">{item.next_step}</div>
                  </div>
                  <div>
                    <div className="cell-label">Top issue</div>
                    <div className="cell-value">{item.top_issue ?? "—"}</div>
                  </div>
                  <div>
                    <div className="cell-label">Keep rate</div>
                    <div className="cell-value">
                      {typeof item.metrics_summary.keep_rate === "number"
                        ? `${Math.round(item.metrics_summary.keep_rate * 100)}%`
                        : "—"}
                    </div>
                  </div>
                  <div>
                    <div className="cell-label">Eval</div>
                    <div className="cell-value">
                      {item.metrics_summary.eval_metric_name
                        ? `${item.metrics_summary.eval_metric_name}: ${item.metrics_summary.eval_metric_value ?? "—"}`
                        : "pending"}
                    </div>
                  </div>
                  <div>
                    <div className="cell-label">Final loss</div>
                    <div className="cell-value">{item.metrics_summary.final_train_loss ?? "—"}</div>
                  </div>
                </div>
              </Link>
            ))
          ) : (
            <EmptyState
              title="No completed runs"
              body="As training summaries are written, this workspace becomes the fastest place to review outcomes and recovery paths."
            />
          )}
        </div>
      </SectionCard>
    </AppShell>
  );
}
