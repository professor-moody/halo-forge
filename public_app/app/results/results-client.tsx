"use client";

import { useState } from "react";

import { apiGet } from "../../lib/api";
import {
  DetailDrawer,
  EmptyState,
  InlineCallout,
  MetricTile,
  ResearchSection,
  SectionCard,
  StatusChip,
} from "../../components/ui";

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
  if (typeof value !== "number") {
    return "—";
  }
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
      <SectionCard
        title="Training outcomes"
        subtitle="Each row should answer whether the run worked, why it behaved that way, and what to do next."
        eyebrow="Outcome review"
      >
        {drawerError ? (
          <InlineCallout title="Detail loading failed" body={drawerError} tone="danger" />
        ) : null}
        <div className="data-table">
          {initialItems.length ? (
            initialItems.map((item) => (
              <button
                key={item.id}
                type="button"
                className="data-row-button"
                onClick={() => void openDetail(item.id)}
                disabled={loadingId === item.id}
              >
                <div className="row-main">
                  <div className="row-title">
                    <h3>{item.modality.toUpperCase()} · {item.model_name}</h3>
                    <p>{item.user_summary.why_it_matters}</p>
                  </div>
                  <StatusChip tone={item.user_summary.confidence_tone} label={item.headline} />
                </div>
                <div className="row-metrics">
                  <div className="row-detail">
                    <div className="cell-label">Top issue</div>
                    <strong>{item.top_issue ?? "—"}</strong>
                  </div>
                  <div className="row-detail">
                    <div className="cell-label">Recommended next step</div>
                    <strong>{item.next_step}</strong>
                  </div>
                  <div className="row-detail">
                    <div className="cell-label">Keep rate</div>
                    <strong>{percent(item.metrics_summary.keep_rate)}</strong>
                  </div>
                  <div className="row-detail">
                    <div className="cell-label">Eval</div>
                    <strong>
                      {item.metrics_summary.eval_metric_name
                        ? `${item.metrics_summary.eval_metric_name}: ${item.metrics_summary.eval_metric_value ?? "—"}`
                        : "pending"}
                    </strong>
                  </div>
                  <div className="row-detail">
                    <div className="cell-label">Final loss</div>
                    <strong>{item.metrics_summary.final_train_loss ?? "—"}</strong>
                  </div>
                </div>
              </button>
            ))
          ) : (
            <EmptyState
              title="No completed runs"
              body="Completed training outcomes will appear here as soon as run summaries are written."
            />
          )}
        </div>
      </SectionCard>

      <DetailDrawer
        open={selected !== null}
        title={selected?.headline ?? "Run details"}
        subtitle={selected ? `${selected.modality.toUpperCase()} · ${selected.model_name}` : undefined}
        onClose={() => setSelected(null)}
      >
        {selected ? (
          <div className="stack-tight">
            <InlineCallout
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
            <div className="metric-grid">
              <MetricTile label="Next step" value={selected.next_step} />
              <MetricTile label="Top issue" value={selected.top_issue ?? "—"} />
              <MetricTile label="Keep rate" value={percent(selected.metrics_summary.keep_rate)} />
              <MetricTile
                label="Eval"
                value={
                  selected.metrics_summary.eval_metric_name
                    ? `${selected.metrics_summary.eval_metric_name}: ${selected.metrics_summary.eval_metric_value ?? "—"}`
                    : "pending"
                }
              />
            </div>
            {selected.recovery.status === "ready" ? (
              <InlineCallout
                title={selected.recovery.recommended_action || "Recovery available"}
                body={selected.recovery.evidence_summary}
                tone="warning"
              />
            ) : null}
            <SectionCard title="Research details" subtitle="Structured evidence, collapsed by default.">
              <div className="stack-tight">
                {selected.research_sections.map((section) => (
                  <ResearchSection key={section.key} title={section.title} summary={section.summary}>
                    <div className="research-items">
                      {section.items.map((item, index) => (
                        <div key={`${section.key}-${index}`} className="research-item">
                          {Object.entries(item).map(([key, value]) => (
                            <div key={key}>
                              <div className="cell-label">{key.replace(/_/g, " ")}</div>
                              <strong>{String(value ?? "—")}</strong>
                            </div>
                          ))}
                        </div>
                      ))}
                    </div>
                  </ResearchSection>
                ))}
              </div>
            </SectionCard>
          </div>
        ) : null}
      </DetailDrawer>
    </>
  );
}
