"use client";

import { useEffect, useState } from "react";

import { apiGet, apiPost } from "../../lib/api";
import {
  ActionLink,
  EmptyState,
  MetricPill,
  SectionCard,
  StatusChip,
} from "../../components/ui";

type Preset = {
  key: string;
  mode: string;
  label: string;
  description: string;
  when_to_use: string;
  expected_runtime: string;
  yield_safety: string;
  values: Record<string, string | number | boolean>;
};

type PreflightResponse = {
  ok: boolean;
  errors: string[];
  warnings: string[];
  suggested_fixes: string[];
  user_summary: {
    headline: string;
    why_it_matters: string;
    confidence_tone: string;
    next_step: string;
  };
  details: {
    recommended_adjustment?: string;
    quality_outlook?: {
      artifact_notes?: string[];
    };
  };
};

export function TrainClient() {
  const [presets, setPresets] = useState<Preset[]>([]);
  const [selectedPreset, setSelectedPreset] = useState<string>("");
  const [form, setForm] = useState<Record<string, string | number | boolean>>({
    mode: "sft",
    model: "Qwen/Qwen2.5-Coder-1.5B",
    dataset: "codealpaca",
    output_dir: "models/sft_public_run",
    epochs: 1,
    batch_size: 2,
    gradient_accumulation_steps: 4,
    max_samples: 200,
  });
  const [preflight, setPreflight] = useState<PreflightResponse | null>(null);
  const [launchError, setLaunchError] = useState<string>("");
  const [launchedRunId, setLaunchedRunId] = useState<string>("");

  useEffect(() => {
    void apiGet<{ items: Preset[] }>("/train/presets").then((payload) => {
      setPresets(payload.items);
      if (payload.items.length && !selectedPreset) {
        const first = payload.items[0];
        setSelectedPreset(first.key);
        setForm({ mode: first.mode, ...first.values });
      }
    });
  }, [selectedPreset]);

  function applyPreset(key: string) {
    const preset = presets.find((item) => item.key === key);
    if (!preset) {
      return;
    }
    setSelectedPreset(key);
    setPreflight(null);
    setLaunchError("");
    setLaunchedRunId("");
    setForm({ mode: preset.mode, ...preset.values });
  }

  function updateField(name: string, value: string | number | boolean) {
    setForm((current) => ({ ...current, [name]: value }));
  }

  async function runPreflight() {
    setLaunchError("");
    const payload = await apiPost<PreflightResponse>("/train/preflight", form);
    setPreflight(payload);
  }

  async function launchRun() {
    try {
      setLaunchError("");
      const payload = await apiPost<{ id: string }>("/train/launch", form);
      setLaunchedRunId(payload.id);
    } catch (error) {
      setLaunchError(error instanceof Error ? error.message : "Launch failed.");
    }
  }

  const mode = String(form.mode ?? "sft");
  const activePreset = presets.find((preset) => preset.key === selectedPreset);
  const datasetField = mode === "raft" ? "prompts" : "dataset";
  const runShapeField = mode === "sft" ? "epochs" : "cycles";
  const runShapeLabel = mode === "sft" ? "Epochs" : "Cycles";
  const budgetField = mode === "sft" ? "max_samples" : mode === "raft" ? "samples_per_prompt" : "limit";
  const budgetLabel =
    mode === "sft" ? "Max samples" : mode === "raft" ? "Samples per prompt" : "Dataset limit";

  return (
    <div className="form-grid">
      <div className="form-sections">
        <SectionCard title="Launch workspace" subtitle="Preset-first control panel with grouped input sections.">
          <div className="form-sections">
            <div className="form-section">
              <h3>Required inputs</h3>
              <div className="field-grid">
                <div className="field">
                  <label>Preset</label>
                  <select value={selectedPreset} onChange={(event) => applyPreset(event.target.value)}>
                    {presets.map((preset) => (
                      <option key={preset.key} value={preset.key}>
                        {preset.label}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field">
                  <label>Mode</label>
                  <select value={mode} onChange={(event) => updateField("mode", event.target.value)}>
                    <option value="sft">SFT</option>
                    <option value="raft">RAFT</option>
                    <option value="vlm">VLM</option>
                    <option value="audio">Audio</option>
                    <option value="reasoning">Reasoning</option>
                    <option value="agentic">Agentic</option>
                  </select>
                </div>
                <div className="field">
                  <label>Model</label>
                  <input
                    value={String(form.model ?? "")}
                    onChange={(event) => updateField("model", event.target.value)}
                  />
                </div>
                <div className="field">
                  <label>{mode === "raft" ? "Prompts" : "Dataset"}</label>
                  <input
                    value={String(form[datasetField] ?? "")}
                    onChange={(event) => updateField(datasetField, event.target.value)}
                  />
                </div>
                <div className="field">
                  <label>Output directory</label>
                  <input
                    value={String(form.output_dir ?? "")}
                    onChange={(event) => updateField("output_dir", event.target.value)}
                  />
                </div>
              </div>
            </div>

            <div className="form-section">
              <h3>Run shape</h3>
              <div className="field-grid compact">
                <div className="field">
                  <label>{runShapeLabel}</label>
                  <input
                    value={String(form[runShapeField] ?? (mode === "sft" ? "1" : "2"))}
                    onChange={(event) => updateField(runShapeField, Number(event.target.value))}
                  />
                </div>
                <div className="field">
                  <label>{budgetLabel}</label>
                  <input
                    value={String(form[budgetField] ?? "")}
                    onChange={(event) => updateField(budgetField, Number(event.target.value))}
                  />
                </div>
                <div className="field">
                  <label>Batch / Keep control</label>
                  <input
                    value={String(
                      mode === "sft"
                        ? form.batch_size ?? "2"
                        : form.keep_percent ?? "0.5",
                    )}
                    onChange={(event) =>
                      updateField(
                        mode === "sft" ? "batch_size" : "keep_percent",
                        mode === "sft" ? Number(event.target.value) : Number(event.target.value),
                      )
                    }
                  />
                </div>
              </div>
            </div>

            <div className="form-section">
              <h3>Quality-sensitive knobs</h3>
              <p>Keep these conservative for the first useful run. Tighten them only after yield stabilizes.</p>
              <div className="field-grid compact">
                {mode === "sft" ? (
                  <>
                    <div className="field">
                      <label>Gradient accumulation</label>
                      <input
                        value={String(form.gradient_accumulation_steps ?? "4")}
                        onChange={(event) => updateField("gradient_accumulation_steps", Number(event.target.value))}
                      />
                    </div>
                    <div className="field">
                      <label>Learning rate</label>
                      <input
                        value={String(form.learning_rate ?? "0.0002")}
                        onChange={(event) => updateField("learning_rate", Number(event.target.value))}
                      />
                    </div>
                  </>
                ) : (
                  <>
                    <div className="field">
                      <label>Reward threshold</label>
                      <input
                        value={String(form.reward_threshold ?? "0.5")}
                        onChange={(event) => updateField("reward_threshold", Number(event.target.value))}
                      />
                    </div>
                    <div className="field">
                      <label>Temperature</label>
                      <input
                        value={String(form.temperature ?? "0.7")}
                        onChange={(event) => updateField("temperature", Number(event.target.value))}
                      />
                    </div>
                  </>
                )}
              </div>
            </div>

            <div className="button-row">
              <button className="secondary-button" onClick={() => void runPreflight()}>
                Review quality
              </button>
              <button className="primary-button" onClick={() => void launchRun()}>
                Launch run
              </button>
            </div>
            {launchError ? <p style={{ color: "var(--danger)", margin: 0 }}>{launchError}</p> : null}
            {launchedRunId ? (
              <div className="callout">
                <h4>Run launched</h4>
                <p>Open the live monitor to follow progress and recovery guidance.</p>
                <div className="button-row" style={{ marginTop: 12 }}>
                  <ActionLink href={`/runs/${encodeURIComponent(launchedRunId)}`} label="Open run monitor" tone="primary" />
                </div>
              </div>
            ) : null}
          </div>
        </SectionCard>
      </div>

      <div className="form-sections">
        <SectionCard title="Launch review" subtitle="What this preset is for and what it is expected to produce.">
          {activePreset ? (
            <div className="stack">
              <div className="callout">
                <h4>{activePreset.label}</h4>
                <p>{activePreset.description}</p>
              </div>
              <div className="metric-grid">
                <MetricPill label="When to use" value={activePreset.when_to_use} />
                <MetricPill label="Expected runtime" value={activePreset.expected_runtime} />
                <MetricPill label="Yield safety" value={activePreset.yield_safety} />
              </div>
            </div>
          ) : (
            <EmptyState title="Preset loading" body="Fetching preset metadata for this training mode." />
          )}
        </SectionCard>

        <SectionCard title="Quality outlook" subtitle="Advisory review before launch.">
          {preflight ? (
            <div className="stack">
              <StatusChip tone={preflight.user_summary.confidence_tone} label={preflight.user_summary.headline} />
              <div className="callout">
                <h4>Why it matters</h4>
                <p>{preflight.user_summary.why_it_matters}</p>
              </div>
              {preflight.details.recommended_adjustment ? (
                <div className="callout">
                  <h4>Recommended adjustment</h4>
                  <p>{preflight.details.recommended_adjustment}</p>
                </div>
              ) : null}
              <div className="metric-grid">
                <MetricPill label="Warnings" value={String(preflight.warnings.length)} />
                <MetricPill label="Fixes" value={String(preflight.suggested_fixes.length)} />
                <MetricPill label="Launch state" value={preflight.ok ? "ready" : "needs fixes"} />
              </div>
              <div className="callout">
                <h4>Expected artifacts</h4>
                <p>
                  {(preflight.details.quality_outlook?.artifact_notes ?? []).join(", ") || "Artifacts unavailable."}
                </p>
              </div>
            </div>
          ) : (
            <EmptyState
              title="No launch review yet"
              body="Review quality to get a compact risk summary before you commit the run."
            />
          )}
        </SectionCard>
      </div>
    </div>
  );
}
