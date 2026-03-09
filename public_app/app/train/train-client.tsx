"use client";

import { useEffect, useState } from "react";

import { apiGet, apiPost } from "../../lib/api";
import {
  ActionLink,
  ActionButton,
  InlineCallout,
  MetricTile,
  SectionCard,
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

function numberInputValue(value: string | number | boolean | undefined) {
  return value === undefined ? "" : String(value);
}

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
  const budgetField =
    mode === "sft" ? "max_samples" : mode === "raft" ? "samples_per_prompt" : "limit";
  const budgetLabel =
    mode === "sft"
      ? "Max samples"
      : mode === "raft"
        ? "Samples per prompt"
        : "Dataset limit";
  const highlightedFields = new Set(
    preflight?.suggested_fixes.some(Boolean)
      ? [mode === "sft" ? "learning_rate" : "reward_threshold"]
      : [],
  );

  return (
    <div className="split-layout">
      <div className="stack-tight">
        <SectionCard
          title="Choose a run profile"
          subtitle="Presets should feel like product modes. Start with one, then tune only what matters."
          eyebrow="Quickstart"
        >
          <div className="segmented-preset-list">
            {presets.map((preset) => (
              <button
                key={preset.key}
                type="button"
                className={
                  preset.key === selectedPreset
                    ? "segmented-preset is-active"
                    : "segmented-preset"
                }
                onClick={() => applyPreset(preset.key)}
              >
                <strong>{preset.label}</strong>
                <small>{preset.description}</small>
              </button>
            ))}
          </div>
        </SectionCard>

        <SectionCard
          title="Run configuration"
          subtitle="The launch controls are grouped by what the user is deciding, not by raw trainer parameters."
          eyebrow="Configuration"
        >
          <div className="stack-tight">
            <div className="config-block">
              <h3>Required inputs</h3>
              <div className="field-grid">
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

            <div className="config-block">
              <h3>Run shape</h3>
              <p>These settings control the amount of work the run will attempt before you inspect the outcome.</p>
              <div className="field-grid compact">
                <div className="field">
                  <label>{runShapeLabel}</label>
                  <input
                    value={numberInputValue(form[runShapeField] ?? (mode === "sft" ? "1" : "2"))}
                    onChange={(event) => updateField(runShapeField, Number(event.target.value))}
                  />
                </div>
                <div className="field">
                  <label>{budgetLabel}</label>
                  <input
                    value={numberInputValue(form[budgetField] ?? "")}
                    onChange={(event) => updateField(budgetField, Number(event.target.value))}
                  />
                </div>
                <div className="field">
                  <label>{mode === "sft" ? "Batch size" : "Keep percent"}</label>
                  <input
                    value={numberInputValue(mode === "sft" ? form.batch_size ?? "2" : form.keep_percent ?? "0.5")}
                    onChange={(event) =>
                      updateField(
                        mode === "sft" ? "batch_size" : "keep_percent",
                        Number(event.target.value),
                      )
                    }
                  />
                </div>
              </div>
            </div>

            <div className="config-block">
              <h3>Quality-sensitive knobs</h3>
              <p>Only change these if the launch review tells you the first configuration is too weak or too strict.</p>
              <div className="field-grid compact">
                {mode === "sft" ? (
                  <>
                    <div className={highlightedFields.has("learning_rate") ? "field is-highlighted" : "field"}>
                      <label>Learning rate</label>
                      <input
                        value={numberInputValue(form.learning_rate ?? "0.0002")}
                        onChange={(event) => updateField("learning_rate", Number(event.target.value))}
                      />
                    </div>
                    <div className="field">
                      <label>Gradient accumulation</label>
                      <input
                        value={numberInputValue(form.gradient_accumulation_steps ?? "4")}
                        onChange={(event) =>
                          updateField("gradient_accumulation_steps", Number(event.target.value))
                        }
                      />
                    </div>
                  </>
                ) : (
                  <>
                    <div className={highlightedFields.has("reward_threshold") ? "field is-highlighted" : "field"}>
                      <label>Reward threshold</label>
                      <input
                        value={numberInputValue(form.reward_threshold ?? "0.5")}
                        onChange={(event) => updateField("reward_threshold", Number(event.target.value))}
                      />
                    </div>
                    <div className="field">
                      <label>Temperature</label>
                      <input
                        value={numberInputValue(form.temperature ?? "0.7")}
                        onChange={(event) => updateField("temperature", Number(event.target.value))}
                      />
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        </SectionCard>
      </div>

      <aside className="review-column">
        <SectionCard
          title="Launch review"
          subtitle="This panel explains what the selected run is for and whether the current configuration looks safe enough to try."
          eyebrow="Run profile"
        >
          <div className="stack-tight">
            <InlineCallout
              title={activePreset?.label ?? "No preset selected"}
              body={activePreset?.description ?? "Choose a preset to load a supported starting configuration."}
              tone="neutral"
            />
            <div className="metric-grid">
              <MetricTile label="Mode" value={mode.toUpperCase()} meta="Training path selected for this run" />
              <MetricTile label="Runtime" value={activePreset?.expected_runtime ?? "unknown"} meta="Expected run size from the preset" />
              <MetricTile label="Yield safety" value={activePreset?.yield_safety ?? "unknown"} meta="How conservative the preset is about data yield" />
            </div>
            <InlineCallout
              title="When to use this"
              body={activePreset?.when_to_use ?? "Preset guidance will appear here when a preset is selected."}
              tone="success"
            />
            <div className="config-block">
              <h3>Expected outputs</h3>
              <div className="field-grid compact">
                <div className="row-detail">
                  <div className="cell-label">Model</div>
                  <strong>{String(form.model ?? "—")}</strong>
                </div>
                <div className="row-detail">
                  <div className="cell-label">Dataset</div>
                  <strong>{String(form[datasetField] ?? "—")}</strong>
                </div>
                <div className="row-detail">
                  <div className="cell-label">Output</div>
                  <strong>{String(form.output_dir ?? "—")}</strong>
                </div>
              </div>
            </div>
          </div>
        </SectionCard>

        <SectionCard
          title="Launch outlook"
          subtitle="Review the current risk summary before you commit the run."
          eyebrow="Recommendation"
          actions={
            <div className="button-row">
              <ActionButton label="Review launch" tone="secondary" onClick={() => void runPreflight()} />
              <ActionButton label="Start run" tone="primary" onClick={() => void launchRun()} />
            </div>
          }
        >
          <div className="stack-tight">
            {preflight ? (
              <>
                <InlineCallout
                  title={preflight.user_summary.headline}
                  body={preflight.user_summary.why_it_matters}
                  tone={
                    preflight.user_summary.confidence_tone === "danger"
                      ? "danger"
                      : preflight.user_summary.confidence_tone === "warning"
                        ? "warning"
                        : "success"
                  }
                />
                <div className="metric-grid">
                  <MetricTile label="Recommended next step" value={preflight.user_summary.next_step} />
                  <MetricTile
                    label="Recommended adjustment"
                    value={preflight.details.recommended_adjustment ?? "None"}
                  />
                </div>
                {preflight.errors.length ? (
                  <InlineCallout
                    title="Errors found"
                    body={preflight.errors.join(" ")}
                    tone="danger"
                  />
                ) : null}
                {preflight.warnings.length ? (
                  <InlineCallout
                    title="Warnings"
                    body={preflight.warnings.join(" ")}
                    tone="warning"
                  />
                ) : null}
                {preflight.suggested_fixes.length ? (
                  <InlineCallout
                    title="Suggested fixes"
                    body={preflight.suggested_fixes.join(" ")}
                    tone="neutral"
                  />
                ) : null}
              </>
            ) : (
              <InlineCallout
                title="No launch review yet"
                body="Run a launch review to see quality risk, recommended next step, and any fixable issues before you start."
                tone="neutral"
              />
            )}

            {launchError ? (
              <InlineCallout title="Launch failed" body={launchError} tone="danger" />
            ) : null}

            {launchedRunId ? (
              <InlineCallout
                title="Run started"
                body="The launch succeeded. Open the run monitor to track progress and recovery guidance."
                tone="success"
                actions={
                  <ActionLink
                    href={`/runs/${encodeURIComponent(launchedRunId)}`}
                    label="Open run monitor"
                    tone="primary"
                  />
                }
              />
            ) : null}
          </div>
        </SectionCard>
      </aside>
    </div>
  );
}
