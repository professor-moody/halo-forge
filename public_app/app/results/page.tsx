import { AppShell, ActionLink } from "../../components/ui";
import { apiGet } from "../../lib/api";
import { ResultsClient } from "./results-client";

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
      subtitle="Review completed runs by outcome, cause, and next action without digging through raw training artifacts."
      statusItems={[
        { label: "Completed runs", value: String(payload.items.length), tone: "neutral" },
        { label: "Review focus", value: "outcomes", tone: "success" },
      ]}
      headerActions={<ActionLink href="/train" label="Start training" tone="primary" />}
    >
      <ResultsClient initialItems={payload.items} />
    </AppShell>
  );
}
