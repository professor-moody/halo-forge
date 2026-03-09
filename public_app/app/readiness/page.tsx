import { ActionLink, AppShell, EmptyState, SectionCard, StatusBadge } from "@/components/app-ui";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { apiGet } from "@/lib/api";

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
      subtitle="Qualification status for each training modality."
      statusItems={[
        {
          label: "Qualification",
          value: payload.aggregate_tier,
          tone: payload.aggregate_tier === "production_ready" ? "success" : payload.aggregate_tier === "qualified" ? "warning" : "neutral",
        },
        { label: "Modalities", value: String(payload.items.length), tone: "neutral" },
      ]}
      headerActions={<ActionLink href="/train" label="Start training" tone="secondary" />}
    >
      <SectionCard title="Qualification matrix" eyebrow="Readiness">
        {payload.items.length ? (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Modality</TableHead>
                <TableHead>Tier</TableHead>
                <TableHead>Metric</TableHead>
                <TableHead className="text-right">Delta</TableHead>
                <TableHead>Next step</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {payload.items.map((item) => (
                <TableRow key={item.modality}>
                  <TableCell>
                    <div className="font-medium">{item.modality.toUpperCase()}</div>
                    <div className="text-xs text-muted-foreground mt-0.5">{item.caveat}</div>
                  </TableCell>
                  <TableCell>
                    <StatusBadge
                      tone={item.production_ready ? "success" : item.readiness_tier === "qualified" ? "warning" : "neutral"}
                      label={item.readiness_tier}
                    />
                  </TableCell>
                  <TableCell className="text-muted-foreground">{item.eval_metric_name || "pending"}</TableCell>
                  <TableCell className="text-right tabular-nums">{item.delta ?? "—"}</TableCell>
                  <TableCell className="text-muted-foreground">{item.next_step}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        ) : (
          <EmptyState title="Readiness unavailable" body="Qualification data could not be loaded." />
        )}
      </SectionCard>
    </AppShell>
  );
}
