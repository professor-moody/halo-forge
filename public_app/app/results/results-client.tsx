"use client";

import { useState } from "react";

import { apiGet } from "@/lib/api";
import {
  Callout,
  EmptyState,
  MetricRow,
  ResearchSection,
  SectionCard,
  StatusBadge,
} from "@/components/app-ui";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";

type ResultItem = {
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
};

type DetailResponse = {
  id: string;
  headline: string;
  next_step: string;
  top_issue?: string | null;
  status: string;
  modality: string;
  model_name: string;
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
    eval_delta?: number | null;
    progress_percent: number;
  };
  recovery: {
    status: string;
    recommended_action: string;
    evidence_summary: string;
  };
  research_sections: Array<{
    key: string;
    title: string;
    summary: string;
    items: Array<Record<string, unknown>>;
  }>;
};

function percent(value?: number | null) {
  if (typeof value !== "number") return "—";
  return `${Math.round(value * 100)}%`;
}

export function ResultsClient({ initialItems }: { initialItems: ResultItem[] }) {
  const [selected, setSelected] = useState<DetailResponse | null>(null);
  const [loadingId, setLoadingId] = useState<string>("");
  const [drawerError, setDrawerError] = useState<string>("");

  async function openDetail(id: string) {
    try {
      setDrawerError("");
      setLoadingId(id);
      const payload = await apiGet<DetailResponse>(`/runs/${encodeURIComponent(id)}?include_research=true`);
      setSelected(payload);
    } catch (error) {
      setDrawerError(error instanceof Error ? error.message : "Unable to load run details.");
    } finally {
      setLoadingId("");
    }
  }

  return (
    <>
      <SectionCard title="Training outcomes" eyebrow="Outcome review">
        {drawerError ? (
          <Callout title="Failed to load details" body={drawerError} tone="danger" />
        ) : null}
        {initialItems.length ? (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Run</TableHead>
                <TableHead>Outcome</TableHead>
                <TableHead>Top issue</TableHead>
                <TableHead>Next step</TableHead>
                <TableHead className="text-right">Keep rate</TableHead>
                <TableHead className="text-right">Eval</TableHead>
                <TableHead className="text-right">Loss</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {initialItems.map((item) => (
                <TableRow
                  key={item.id}
                  className="cursor-pointer"
                  onClick={() => void openDetail(item.id)}
                  data-loading={loadingId === item.id ? "true" : undefined}
                >
                  <TableCell>
                    <span className="font-medium">{item.modality.toUpperCase()}</span>
                    <span className="text-muted-foreground"> · {item.model_name}</span>
                  </TableCell>
                  <TableCell>
                    <StatusBadge tone={item.user_summary.confidence_tone} label={item.headline} />
                  </TableCell>
                  <TableCell className="text-muted-foreground">{item.top_issue ?? "—"}</TableCell>
                  <TableCell className="text-muted-foreground">{item.next_step}</TableCell>
                  <TableCell className="text-right tabular-nums">{percent(item.metrics_summary.keep_rate)}</TableCell>
                  <TableCell className="text-right tabular-nums">
                    {item.metrics_summary.eval_metric_name
                      ? `${item.metrics_summary.eval_metric_name}: ${item.metrics_summary.eval_metric_value ?? "—"}`
                      : "—"}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{item.metrics_summary.final_train_loss ?? "—"}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        ) : (
          <EmptyState title="No completed runs" body="Outcomes appear here after run summaries are written." />
        )}
      </SectionCard>

      <Sheet open={selected !== null} onOpenChange={(open) => { if (!open) setSelected(null); }}>
        <SheetContent className="overflow-y-auto p-6">
          {selected ? (
            <>
              <SheetHeader className="p-0 mb-4">
                <div className="text-xs font-medium text-muted-foreground">Run details</div>
                <SheetTitle className="text-base">{selected.headline}</SheetTitle>
                <SheetDescription>
                  {selected.modality.toUpperCase()} · {selected.model_name}
                </SheetDescription>
              </SheetHeader>

              <div className="space-y-4">
                <Callout
                  title={selected.headline}
                  body={selected.user_summary.why_it_matters}
                  tone={
                    selected.user_summary.confidence_tone === "danger"
                      ? "danger"
                      : selected.user_summary.confidence_tone === "warning"
                        ? "warning"
                        : "success"
                  }
                />

                <div className="rounded-md border border-border divide-y divide-border">
                  <MetricRow label="Next step" value={selected.next_step} />
                  <MetricRow label="Top issue" value={selected.top_issue ?? "—"} />
                  <MetricRow label="Keep rate" value={percent(selected.metrics_summary.keep_rate)} />
                  <MetricRow
                    label="Eval"
                    value={
                      selected.metrics_summary.eval_metric_name
                        ? `${selected.metrics_summary.eval_metric_name}: ${selected.metrics_summary.eval_metric_value ?? "—"}`
                        : "pending"
                    }
                  />
                </div>

                {selected.recovery.status === "ready" ? (
                  <Callout
                    title={selected.recovery.recommended_action || "Recovery available"}
                    body={selected.recovery.evidence_summary}
                    tone="warning"
                  />
                ) : null}

                {selected.research_sections.length > 0 && (
                  <div>
                    <h3 className="text-sm font-medium text-foreground mb-2">Research details</h3>
                    <div className="space-y-2">
                      {selected.research_sections.map((section) => (
                        <ResearchSection key={section.key} title={section.title} summary={section.summary}>
                          <div className="space-y-2 mt-2">
                            {section.items.map((item, index) => (
                              <div key={`${section.key}-${index}`} className="rounded-md border border-border p-2 space-y-1">
                                {Object.entries(item).map(([key, value]) => (
                                  <div key={key} className="flex items-baseline justify-between text-xs">
                                    <span className="text-muted-foreground">{key.replace(/_/g, " ")}</span>
                                    <span className="font-medium text-foreground">{String(value ?? "—")}</span>
                                  </div>
                                ))}
                              </div>
                            ))}
                          </div>
                        </ResearchSection>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </>
          ) : null}
        </SheetContent>
      </Sheet>
    </>
  );
}
