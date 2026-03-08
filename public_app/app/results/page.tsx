import Link from "next/link";

import { AppShell, SectionCard, StatusChip } from "../../components/ui";
import { apiGet } from "../../lib/api";

type ResultsResponse = {
  items: Array<{
    id: string;
    modality: string;
    model_name: string;
    timestamp: string;
    user_summary: {
      headline: string;
      why_it_matters: string;
      next_step: string;
      confidence_tone: string;
    };
    details: {
      verdict?: string | null;
      keep_rate?: number | null;
      top_issue?: string | null;
      final_train_loss?: number | null;
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
    <AppShell>
      <SectionCard
        title="Results and recovery"
        subtitle="Did the run work, why or why not, and what should happen next?"
      >
        <div className="list">
          {payload.items.length ? (
            payload.items.map((item) => (
              <Link key={item.id} href={`/runs/${encodeURIComponent(item.id)}`} className="list-row">
                <header>
                  <div>
                    <h3>{item.modality.toUpperCase()}</h3>
                    <p>{item.model_name}</p>
                  </div>
                  <StatusChip tone={item.user_summary.confidence_tone} label={item.user_summary.headline} />
                </header>
                <p>{item.user_summary.why_it_matters}</p>
                <div className="metric-grid">
                  <div className="metric">
                    <label>Next step</label>
                    <strong>{item.user_summary.next_step}</strong>
                  </div>
                  <div className="metric">
                    <label>Top issue</label>
                    <strong>{item.details.top_issue ?? "—"}</strong>
                  </div>
                  <div className="metric">
                    <label>Keep rate</label>
                    <strong>
                      {typeof item.details.keep_rate === "number"
                        ? `${Math.round(item.details.keep_rate * 100)}%`
                        : "—"}
                    </strong>
                  </div>
                </div>
              </Link>
            ))
          ) : (
            <div className="callout">
              <h3>No completed runs yet</h3>
              <p>Training results will appear here once the first run writes its summary artifact.</p>
            </div>
          )}
        </div>
      </SectionCard>
    </AppShell>
  );
}
