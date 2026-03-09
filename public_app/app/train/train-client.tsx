"use client";

import { useEffect, useState } from "react";

import { apiGet, apiPost } from "@/lib/api";
import { cn } from "@/lib/utils";
import {
  ActionLink,
  ActionButton,
  Callout,
  MetricRow,
  SectionCard,
} from "@/components/app-ui";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

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
    if (!preset) return;
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
        ? "Samples / prompt"
        : "Dataset limit";
  const highlightedFields = new Set(
    preflight?.suggested_fixes.some(Boolean)
      ? [mode === "sft" ? "learning_rate" : "reward_threshold"]
      : [],
  );

  return (
    <div className="grid grid-cols-[minmax(0,1.4fr)_minmax(320px,0.8fr)] gap-4 items-start">
      <div className="space-y-3">
        <SectionCard title="Run profile" eyebrow="Quickstart">
          <div className="flex flex-wrap gap-1.5">
            {presets.map((preset) => (
              <button
                key={preset.key}
                type="button"
                className={cn(
                  "px-3 py-1.5 rounded-md text-sm text-left border transition-colors",
                  preset.key === selectedPreset
                    ? "bg-accent border-primary/40 text-foreground font-medium"
                    : "border-border text-muted-foreground hover:text-foreground hover:bg-accent/50",
                )}
                onClick={() => applyPreset(preset.key)}
              >
                <span className="font-medium text-foreground">{preset.label}</span>
                <span className="block text-xs text-muted-foreground mt-0.5">{preset.description}</span>
              </button>
            ))}
          </div>
        </SectionCard>

        <Card>
          <CardHeader className="pb-3">
            <div className="text-xs font-medium text-muted-foreground">Configuration</div>
            <CardTitle className="text-sm">Run configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <h3 className="text-sm font-medium text-foreground mb-2">Required inputs</h3>
              <div className="grid grid-cols-2 gap-3">
                <FieldGroup label="Mode">
                  <select
                    value={mode}
                    onChange={(e) => updateField("mode", e.target.value)}
                    className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                  >
                    <option value="sft">SFT</option>
                    <option value="raft">RAFT</option>
                    <option value="vlm">VLM</option>
                    <option value="audio">Audio</option>
                    <option value="reasoning">Reasoning</option>
                    <option value="agentic">Agentic</option>
                  </select>
                </FieldGroup>
                <FieldGroup label="Model">
                  <Input
                    value={String(form.model ?? "")}
                    onChange={(e) => updateField("model", e.target.value)}
                  />
                </FieldGroup>
                <FieldGroup label={mode === "raft" ? "Prompts" : "Dataset"}>
                  <Input
                    value={String(form[datasetField] ?? "")}
                    onChange={(e) => updateField(datasetField, e.target.value)}
                  />
                </FieldGroup>
                <FieldGroup label="Output directory">
                  <Input
                    value={String(form.output_dir ?? "")}
                    onChange={(e) => updateField("output_dir", e.target.value)}
                  />
                </FieldGroup>
              </div>
            </div>

            <div>
              <h3 className="text-sm font-medium text-foreground mb-1">Run shape</h3>
              <p className="text-xs text-muted-foreground mb-2">Controls the amount of work before inspection.</p>
              <div className="grid grid-cols-3 gap-3">
                <FieldGroup label={runShapeLabel}>
                  <Input
                    value={numberInputValue(form[runShapeField] ?? (mode === "sft" ? "1" : "2"))}
                    onChange={(e) => updateField(runShapeField, Number(e.target.value))}
                  />
                </FieldGroup>
                <FieldGroup label={budgetLabel}>
                  <Input
                    value={numberInputValue(form[budgetField] ?? "")}
                    onChange={(e) => updateField(budgetField, Number(e.target.value))}
                  />
                </FieldGroup>
                <FieldGroup label={mode === "sft" ? "Batch size" : "Keep percent"}>
                  <Input
                    value={numberInputValue(mode === "sft" ? form.batch_size ?? "2" : form.keep_percent ?? "0.5")}
                    onChange={(e) =>
                      updateField(mode === "sft" ? "batch_size" : "keep_percent", Number(e.target.value))
                    }
                  />
                </FieldGroup>
              </div>
            </div>

            <div>
              <h3 className="text-sm font-medium text-foreground mb-1">Quality-sensitive knobs</h3>
              <p className="text-xs text-muted-foreground mb-2">Only change if the launch review recommends it.</p>
              <div className="grid grid-cols-2 gap-3">
                {mode === "sft" ? (
                  <>
                    <FieldGroup
                      label="Learning rate"
                      highlighted={highlightedFields.has("learning_rate")}
                    >
                      <Input
                        value={numberInputValue(form.learning_rate ?? "0.0002")}
                        onChange={(e) => updateField("learning_rate", Number(e.target.value))}
                      />
                    </FieldGroup>
                    <FieldGroup label="Gradient accumulation">
                      <Input
                        value={numberInputValue(form.gradient_accumulation_steps ?? "4")}
                        onChange={(e) => updateField("gradient_accumulation_steps", Number(e.target.value))}
                      />
                    </FieldGroup>
                  </>
                ) : (
                  <>
                    <FieldGroup
                      label="Reward threshold"
                      highlighted={highlightedFields.has("reward_threshold")}
                    >
                      <Input
                        value={numberInputValue(form.reward_threshold ?? "0.5")}
                        onChange={(e) => updateField("reward_threshold", Number(e.target.value))}
                      />
                    </FieldGroup>
                    <FieldGroup label="Temperature">
                      <Input
                        value={numberInputValue(form.temperature ?? "0.7")}
                        onChange={(e) => updateField("temperature", Number(e.target.value))}
                      />
                    </FieldGroup>
                  </>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <aside className="space-y-3 sticky top-4">
        <Card>
          <CardHeader className="pb-3">
            <div className="text-xs font-medium text-muted-foreground">Run profile</div>
            <CardTitle className="text-sm">Launch review</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <Callout
              title={activePreset?.label ?? "No preset selected"}
              body={activePreset?.description ?? "Choose a preset to load a starting configuration."}
              tone="neutral"
            />
            <div className="rounded-md border border-border divide-y divide-border">
              <MetricRow label="Mode" value={mode.toUpperCase()} />
              <MetricRow label="Runtime" value={activePreset?.expected_runtime ?? "unknown"} />
              <MetricRow label="Yield safety" value={activePreset?.yield_safety ?? "unknown"} />
            </div>
            <Callout
              title="When to use"
              body={activePreset?.when_to_use ?? "Preset guidance appears when a preset is selected."}
              tone="success"
            />
            <div className="rounded-md border border-border divide-y divide-border">
              <MetricRow label="Model" value={String(form.model ?? "—")} />
              <MetricRow label="Dataset" value={String(form[datasetField] ?? "—")} />
              <MetricRow label="Output" value={String(form.output_dir ?? "—")} />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-start justify-between">
              <div>
                <div className="text-xs font-medium text-muted-foreground">Recommendation</div>
                <CardTitle className="text-sm mt-1">Launch outlook</CardTitle>
              </div>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" onClick={() => void runPreflight()}>
                  Review
                </Button>
                <Button size="sm" onClick={() => void launchRun()}>
                  Start run
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            {preflight ? (
              <>
                <Callout
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
                <div className="rounded-md border border-border divide-y divide-border">
                  <MetricRow label="Next step" value={preflight.user_summary.next_step} />
                  <MetricRow label="Adjustment" value={preflight.details.recommended_adjustment ?? "None"} />
                </div>
                {preflight.errors.length ? (
                  <Callout title="Errors" body={preflight.errors.join(" ")} tone="danger" />
                ) : null}
                {preflight.warnings.length ? (
                  <Callout title="Warnings" body={preflight.warnings.join(" ")} tone="warning" />
                ) : null}
                {preflight.suggested_fixes.length ? (
                  <Callout title="Suggested fixes" body={preflight.suggested_fixes.join(" ")} tone="neutral" />
                ) : null}
              </>
            ) : (
              <Callout
                title="No review yet"
                body="Run a launch review to see risk, next step, and fixable issues."
                tone="neutral"
              />
            )}

            {launchError ? (
              <Callout title="Launch failed" body={launchError} tone="danger" />
            ) : null}

            {launchedRunId ? (
              <Callout
                title="Run started"
                body="Launch succeeded. Open the run monitor to track progress."
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
          </CardContent>
        </Card>
      </aside>
    </div>
  );
}

function FieldGroup({
  label,
  highlighted,
  children,
}: {
  label: string;
  highlighted?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1">
      <label className={cn("text-xs font-medium", highlighted ? "text-primary" : "text-muted-foreground")}>
        {label}
      </label>
      {children}
    </div>
  );
}
