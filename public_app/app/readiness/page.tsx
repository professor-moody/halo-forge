import { ActionLink, AppShell, EmptyState, SectionCard, StatusChip } from "../../components/ui";
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
      subtitle="Qualification status for each training modality, grounded in deterministic launch, update, artifact, resume, and eval checks."
      statusItems={[
        { label: "Qualification status", value: payload.aggregate_tier, tone: payload.aggregate_tier === "production_ready" ? "success" : payload.aggregate_tier === "qualified" ? "warning" : "neutral" },
        { label: "Modalities", value: String(payload.items.length), tone: "neutral" },
      ]}
      headerActions={<ActionLink href="/train" label="Start training" tone="secondary" />}
    >
      <SectionCard title="Qualification matrix" subtitle="Production-ready means the deterministic train, resume, artifact, and eval checks are currently passing." eyebrow="Readiness">
        {payload.items.length ? (
          <div className="table-matrix">
            <div className="matrix-head">
              <div className="cell-label">Modality</div>
              <div className="cell-label">Tier</div>
              <div className="cell-label">Metric</div>
              <div className="cell-label">Delta</div>
              <div className="cell-label">Next required work</div>
            </div>
            {payload.items.map((item) => (
              <div key={item.modality} className="matrix-row">
                <div className="matrix-primary">
                  <h3>{item.modality.toUpperCase()}</h3>
                  <p>{item.caveat}</p>
                </div>
                <div>
                  <StatusChip tone={item.production_ready ? "success" : item.readiness_tier === "qualified" ? "warning" : "neutral"} label={item.readiness_tier} />
                </div>
                <div className="row-detail">
                  <div className="cell-label">Metric</div>
                  <strong>{item.eval_metric_name || "pending"}</strong>
                </div>
                <div className="row-detail">
                  <div className="cell-label">Delta</div>
                  <strong>{item.delta ?? "—"}</strong>
                </div>
                <div className="row-detail">
                  <div className="cell-label">Next step</div>
                  <strong>{item.next_step}</strong>
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
