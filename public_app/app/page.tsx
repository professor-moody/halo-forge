import Link from "next/link";

import { AppShell, SectionCard, StatusChip } from "../components/ui";
import { apiGet } from "../lib/api";

type ReadinessResponse = {
  aggregate_tier: string;
  items: Array<{ modality: string; readiness_tier: string; production_ready: boolean }>;
};

type RunsResponse = {
  items: Array<{
    id: string;
    modality: string;
    model_name: string;
    user_summary: { headline: string; why_it_matters: string; confidence_tone: string };
  }>;
};

async function getDashboardData() {
  try {
    const [readiness, runs] = await Promise.all([
      apiGet<ReadinessResponse>("/readiness"),
      apiGet<RunsResponse>("/runs?include_completed=true&active_only=false&include_research=false"),
    ]);
    return { readiness, runs };
  } catch {
    return { readiness: null, runs: null };
  }
}

export default async function HomePage() {
  const { readiness, runs } = await getDashboardData();
  const recent = runs?.items.slice(0, 3) ?? [];

  return (
    <AppShell>
      <SectionCard
        title="Public training workspace"
        subtitle="Clear launch paths for users, deeper evidence when researchers need it."
      >
        <div className="grid-two">
          <div className="callout">
            <h3>Start from one strong workflow</h3>
            <p>
              Launch training, follow live quality, and recover from weak runs without
              starting in an operator console.
            </p>
            <div className="button-row" style={{ marginTop: 14 }}>
              <Link className="primary-button" href="/train">
                Open training
              </Link>
              <Link className="secondary-button" href="/results">
                Review results
              </Link>
            </div>
          </div>
          <div className="callout">
            <h3>Readiness snapshot</h3>
            <p>
              Qualification truth stays in the backend and drives the public labels shown here.
            </p>
            <div style={{ marginTop: 12 }}>
              <StatusChip
                tone={
                  readiness?.aggregate_tier === "production_ready"
                    ? "success"
                    : readiness?.aggregate_tier === "qualified"
                      ? "warning"
                      : "neutral"
                }
                label={readiness?.aggregate_tier ?? "readiness unavailable"}
              />
            </div>
          </div>
        </div>
      </SectionCard>

      <SectionCard title="Recent runs" subtitle="Outcome first, diagnostics second.">
        <div className="list">
          {recent.length ? (
            recent.map((run) => (
              <Link key={run.id} href={`/runs/${encodeURIComponent(run.id)}`} className="list-row">
                <header>
                  <div>
                    <h3>{run.modality.toUpperCase()}</h3>
                    <p>{run.model_name}</p>
                  </div>
                  <StatusChip tone={run.user_summary.confidence_tone} label={run.user_summary.headline} />
                </header>
                <p>{run.user_summary.why_it_matters}</p>
              </Link>
            ))
          ) : (
            <div className="callout">
              <h3>No training runs detected yet</h3>
              <p>Start with a quickstart preset and the public surface will populate itself.</p>
            </div>
          )}
        </div>
      </SectionCard>

      <SectionCard title="What changed" subtitle="The public surface is intentionally narrower than the internal console.">
        <div className="metric-grid">
          <div className="metric">
            <label>Public default</label>
            <strong>Plain-language next step</strong>
          </div>
          <div className="metric">
            <label>Research details</label>
            <strong>Expandable, not dominant</strong>
          </div>
          <div className="metric">
            <label>Internal tooling</label>
            <strong>Still in the NiceGUI console</strong>
          </div>
        </div>
      </SectionCard>
    </AppShell>
  );
}
