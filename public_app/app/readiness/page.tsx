import { AppShell, SectionCard, StatusChip } from "../../components/ui";
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
    <AppShell>
      <SectionCard
        title="Training modality readiness"
        subtitle="Qualification truth is public here, but launch access stays broader than production-ready status."
      >
        <div className="stack">
          <StatusChip
            tone={
              payload.aggregate_tier === "production_ready"
                ? "success"
                : payload.aggregate_tier === "qualified"
                  ? "warning"
                  : "neutral"
            }
            label={payload.aggregate_tier}
          />
          <div className="list">
            {payload.items.map((item) => (
              <div key={item.modality} className="list-row">
                <header>
                  <div>
                    <h3>{item.modality.toUpperCase()}</h3>
                    <p>{item.eval_metric_name || "eval metric pending"}</p>
                  </div>
                  <StatusChip
                    tone={item.production_ready ? "success" : item.readiness_tier === "qualified" ? "warning" : "neutral"}
                    label={item.readiness_tier}
                  />
                </header>
                <p>{item.caveat}</p>
                <div className="metric-grid">
                  <div className="metric">
                    <label>Next step</label>
                    <strong>{item.next_step}</strong>
                  </div>
                  <div className="metric">
                    <label>Baseline</label>
                    <strong>{item.baseline_value ?? "—"}</strong>
                  </div>
                  <div className="metric">
                    <label>Current</label>
                    <strong>{item.final_value ?? "—"}</strong>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </SectionCard>
    </AppShell>
  );
}
