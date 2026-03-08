"use client";

import { useEffect, useState } from "react";

import { apiGet, apiPost } from "../../lib/api";
import { SectionCard, StatusChip } from "../../components/ui";

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
  };
  details: {
    recommended_adjustment?: string;
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
    setForm({ mode: preset.mode, ...preset.values });
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

  return (
    <div className="stack">
      <SectionCard
        title="Launch training"
        subtitle="Quickstart first, advanced knobs only when you need them."
      >
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
            <select
              value={mode}
              onChange={(event) => setForm((current) => ({ ...current, mode: event.target.value }))}
            >
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
              onChange={(event) => setForm((current) => ({ ...current, model: event.target.value }))}
            />
          </div>
          <div className="field">
            <label>{mode === "raft" ? "Prompts" : "Dataset"}</label>
            <input
              value={String(mode === "raft" ? form.prompts ?? "" : form.dataset ?? "")}
              onChange={(event) =>
                setForm((current) => ({
                  ...current,
                  [mode === "raft" ? "prompts" : "dataset"]: event.target.value,
                }))
              }
            />
          </div>
          <div className="field">
            <label>Output directory</label>
            <input
              value={String(form.output_dir ?? "")}
              onChange={(event) => setForm((current) => ({ ...current, output_dir: event.target.value }))}
            />
          </div>
          <div className="field">
            <label>{mode === "sft" ? "Max samples" : "Cycles"}</label>
            <input
              value={String(mode === "sft" ? form.max_samples ?? "" : form.cycles ?? "2")}
              onChange={(event) =>
                setForm((current) => ({
                  ...current,
                  [mode === "sft" ? "max_samples" : "cycles"]: Number(event.target.value),
                }))
              }
            />
          </div>
        </div>

        <div className="button-row" style={{ marginTop: 18 }}>
          <button className="secondary-button" onClick={() => void runPreflight()}>
            Review quality outlook
          </button>
          <button className="primary-button" onClick={() => void launchRun()}>
            Launch run
          </button>
        </div>
        {launchError ? <p style={{ color: "var(--danger)" }}>{launchError}</p> : null}
        {launchedRunId ? (
          <p>
            Run launched. Open <a href={`/runs/${encodeURIComponent(launchedRunId)}`}>live monitor</a>.
          </p>
        ) : null}
      </SectionCard>

      <SectionCard title="Training quality outlook" subtitle="Advisory only. Launch blockers stay limited to invalid input.">
        {preflight ? (
          <div className="stack">
            <StatusChip tone={preflight.user_summary.confidence_tone} label={preflight.user_summary.headline} />
            <p>{preflight.user_summary.why_it_matters}</p>
            {preflight.details.recommended_adjustment ? (
              <div className="callout">
                <h4>Recommended adjustment</h4>
                <p>{preflight.details.recommended_adjustment}</p>
              </div>
            ) : null}
            <div className="grid-two">
              <div className="callout">
                <h4>Warnings</h4>
                <ul>
                  {(preflight.warnings.length ? preflight.warnings : ["No major quality warnings detected."]).map(
                    (warning) => (
                      <li key={warning}>{warning}</li>
                    ),
                  )}
                </ul>
              </div>
              <div className="callout">
                <h4>Fixes</h4>
                <ul>
                  {(preflight.suggested_fixes.length
                    ? preflight.suggested_fixes
                    : ["Current settings look balanced for a first pass."]).map((suggestion) => (
                    <li key={suggestion}>{suggestion}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        ) : (
          <p>Run preflight to get a user-friendly quality outlook before launch.</p>
        )}
      </SectionCard>
    </div>
  );
}
