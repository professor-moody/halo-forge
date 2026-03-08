import { AppShell, EmptyState, SectionCard, StatusChip } from "../../components/ui";
import { apiGet } from "../../lib/api";

type ReadinessResponse = {
  aggregate_tier: string;
  items: Array<{
    modality: string;
    readiness_tier: string;
    production_ready: boolean;
    caveat: string;
    next_step: string;
    eval_metric_name: string;
    final_value: number | null;
    baseline_value: number | null;
    delta: number | null;
  }>;
};

async function getReadiness() {
  try {
    return await apiGet<ReadinessResponse>("/readiness");
  } catch {
    return { aggregate_tier: "unavailable", items: [] };
  }
}

export default async function ReadinessPage() {
  const payload = await getReadiness();
  return (
    <AppShell
      title="Readiness"
      subtitle="Qualification-driven matrix for training modalities, with caveats and the next required work."
      statusItems={[
        { label: "Aggregate", value: payload.aggregate_tier, tone: payload.aggregate_tier === "production_ready" ? "success" : payload.aggregate_tier === "qualified" ? "warning" : "neutral" },
        { label: "Modalities", value: String(payload.items.length), tone: "neutral" },
      ]}
    >
      <SectionCard title="Qualification matrix" subtitle="Production-ready means the deterministic train, resume, artifact, and eval checks are currently passing.">
        {payload.items.length ? (
          <div className="run-list">
            {payload.items.map((item) => (
              <div key={item.modality} className="table-row">
                <div className="table-row-header">
                  <div className="table-row-title">
                    <h3>{item.modality.toUpperCase()}</h3>
                    <p>{item.caveat}</p>
                  </div>
                  <StatusChip tone={item.production_ready ? "success" : item.readiness_tier === "qualified" ? "warning" : "neutral"} label={item.readiness_tier} />
                </div>
                <div className="table-row-metrics">
                  <div>
                    <div className="cell-label">Metric</div>
                    <div className="cell-value">{item.eval_metric_name || "pending"}</div>
                  </div>
                  <div>
                    <div className="cell-label">Baseline</div>
                    <div className="cell-value">{item.baseline_value ?? "—"}</div>
                  </div>
                  <div>
                    <div className="cell-label">Current</div>
                    <div className="cell-value">{item.final_value ?? "—"}</div>
                  </div>
                  <div>
                    <div className="cell-label">Delta</div>
                    <div className="cell-value">{item.delta ?? "—"}</div>
                  </div>
                  <div>
                    <div className="cell-label">Next work</div>
                    <div className="cell-value">{item.next_step}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <EmptyState
            title="Readiness unavailable"
            body="Qualification data could not be loaded for the current environment."
          />
        )}
      </SectionCard>
    </AppShell>
  );
}
